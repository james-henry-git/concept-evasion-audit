#!/usr/bin/env python3
"""
B1: Extract CAZ profiles for Neural Chameleons' 11 concepts.

Finds:
  - Separation S(l): Fisher-normalized class distance at each layer
  - Coherence C(l): concentration of representation along a single direction
  - Velocity v(l): dS/dl — rate of change of separation
  - Peak layer: argmax S(l)
  - Assembly zone: layers where S > 0.5 * max(S)
  - Handoff boundary: steepest velocity crossing (transition layer)

Output per concept:
  - Full layer profiles (S, C, v)
  - Peak layer index
  - Assembly zone [start, end]
  - Handoff layer

This is the core Rosetta methodology applied to their concept set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR
from cea.data import BENIGN_CONCEPTS, ALL_CONCEPTS
from cea.extraction import load_model_and_tokenizer, extract_hidden_states

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--concepts", nargs="+", default=BENIGN_CONCEPTS)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--data-file", default=None)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
out_dir = RESULTS_DIR / "B_caz" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

model, tokenizer = load_model_and_tokenizer(args.model)
n_layers = model.config.num_hidden_layers + 1
print(f"Model: {args.model} | Layers: {n_layers}")


def fisher_separation(h_pos: np.ndarray, h_neg: np.ndarray) -> float:
    """Fisher-normalized class separation."""
    mu_pos = h_pos.mean(axis=0)
    mu_neg = h_neg.mean(axis=0)
    var_pos = h_pos.var(axis=0).mean()
    var_neg = h_neg.var(axis=0).mean()
    pooled_var = 0.5 * (var_pos + var_neg) + 1e-8
    diff = mu_pos - mu_neg
    return float(np.dot(diff, diff) / pooled_var)


def coherence(h: np.ndarray) -> float:
    """Fraction of variance captured by first PC."""
    if h.shape[0] < 2:
        return 0.0
    h_centered = h - h.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(h_centered, full_matrices=False)
        total = (s ** 2).sum() + 1e-8
        return float((s[0] ** 2) / total)
    except np.linalg.LinAlgError:
        return 0.0


def find_assembly_zone(separations: np.ndarray, threshold: float = 0.5) -> tuple[int, int]:
    """Layers where S > threshold * max(S)."""
    peak = separations.max()
    above = np.where(separations > threshold * peak)[0]
    if len(above) == 0:
        peak_idx = int(separations.argmax())
        return peak_idx, peak_idx
    return int(above[0]), int(above[-1])


def find_handoff_layer(separations: np.ndarray) -> int:
    """Layer with maximum positive velocity (steepest assembly onset)."""
    velocity = np.gradient(separations)
    return int(velocity.argmax())


results = {}
for concept in args.concepts:
    if concept not in all_data:
        print(f"  Skipping {concept}: not in data")
        continue
    print(f"  Processing: {concept}")

    positives = all_data[concept]["positive"]
    negatives = all_data[concept]["negative"]
    texts = positives + negatives
    n_pos = len(positives)

    # Extract all layers at once
    reps = extract_hidden_states(
        texts, model, tokenizer, batch_size=args.batch_size
    )  # (N, n_layers, D)

    separations = np.zeros(n_layers)
    coherences = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        h = reps[:, layer_idx, :]   # (N, D)
        h_pos = h[:n_pos]
        h_neg = h[n_pos:]
        separations[layer_idx] = fisher_separation(h_pos, h_neg)
        coherences[layer_idx] = coherence(h)

    velocity = np.gradient(separations).tolist()
    peak_layer = int(separations.argmax())
    zone_start, zone_end = find_assembly_zone(separations)
    handoff_layer = find_handoff_layer(separations)

    # Depth-normalized positions (fraction of total depth)
    peak_depth = peak_layer / (n_layers - 1)
    handoff_depth = handoff_layer / (n_layers - 1)
    zone_start_depth = zone_start / (n_layers - 1)
    zone_end_depth = zone_end / (n_layers - 1)

    result = {
        "concept": concept,
        "n_layers": n_layers,
        "separations": separations.tolist(),
        "coherences": coherences.tolist(),
        "velocity": velocity,
        "peak_layer": peak_layer,
        "peak_separation": float(separations[peak_layer]),
        "peak_depth_normalized": peak_depth,
        "handoff_layer": handoff_layer,
        "handoff_depth_normalized": handoff_depth,
        "assembly_zone": {"start": zone_start, "end": zone_end,
                          "start_depth": zone_start_depth, "end_depth": zone_end_depth},
    }
    results[concept] = result

    print(f"    Peak layer: {peak_layer} (depth={peak_depth:.2f}) | "
          f"S={separations[peak_layer]:.3f} | "
          f"Zone: [{zone_start},{zone_end}] | "
          f"Handoff: {handoff_layer} (depth={handoff_depth:.2f})")

    # Save per-concept
    out_file = out_dir / f"{concept}_caz.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

# Save aggregate
agg_file = out_dir / "caz_profiles.json"
with open(agg_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved CAZ profiles → {agg_file}")

# Summary table
print("\n=== CAZ SUMMARY ===")
print(f"{'Concept':15s} {'Peak':>6s} {'Depth':>6s} {'S_peak':>8s} {'Handoff':>8s} {'Zone':>12s}")
print("-" * 60)
for concept, r in sorted(results.items(), key=lambda x: x[1]["peak_depth_normalized"]):
    z = r["assembly_zone"]
    print(f"{concept:15s} {r['peak_layer']:>6d} {r['peak_depth_normalized']:>6.2f} "
          f"{r['peak_separation']:>8.3f} {r['handoff_depth_normalized']:>8.2f} "
          f"[{z['start']},{z['end']}]")

#!/usr/bin/env python3
"""
C3: Per-layer concept-direction cosine profile against the abliteration direction.

For each RCP concept, computes the mean-difference direction at every layer
(d_L = mean(pos_acts[L]) - mean(neg_acts[L]), normalised) and measures its
cosine with the C1 abliteration direction d.

Arditi et al. apply a SINGLE fixed d to all 42 layers. If d aligns with any
concept direction at a layer, that layer's surgery also affects that concept.
This script maps the full collateral damage footprint.

Outputs (in RESULTS_DIR/C_per_layer_cosine/):
  per_layer_cosine.json      — concept × layer cosine matrix
  summary.json               — top hits per concept, peak cosines
  heatmap.png                — concept × layer heatmap (if matplotlib available)

Usage:
    # With C1 direction from Gemma-IT model (canonical):
    python C3_per_layer_concept_cosine.py \\
        --direction ~/rosetta_data/results/concept_evasion/C_weight_diff/... \\
        --alllayer-dir ~/rosetta_data/paper_n250/google_gemma_2_9b_it

    # Quick proxy run using Gemma base alllayer files (available now):
    python C3_per_layer_concept_cosine.py --proxy

    # Or point both explicitly:
    python C3_per_layer_concept_cosine.py \\
        --direction-file <path>/refusal_direction.npy \\
        --alllayer-dir ~/rosetta_data/paper_n250/google_gemma_2_9b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--direction-file", default=None,
                    help="Path to refusal_direction.npy from C1")
parser.add_argument("--alllayer-dir", default=None,
                    help="Dir containing calibration_alllayer_{concept}.npy files")
parser.add_argument("--proxy", action="store_true",
                    help="Quick run: use base Gemma alllayer + best available direction")
parser.add_argument("--model-pair", default=None,
                    help="Which C_weight_diff model pair to use (default: auto-detect)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path.home() / "rosetta_data"

if args.proxy:
    # Use base Gemma (locally available alllayer files)
    alllayer_dir = DATA_ROOT / "paper_n250" / "google_gemma_2_9b"
    # Try to find any available direction
    cwdir = DATA_ROOT / "results" / "concept_evasion" / "C_weight_diff"
    direction_file = None
    if cwdir.exists():
        for pair_dir in sorted(cwdir.iterdir()):
            candidate = pair_dir / "refusal_direction.npy"
            if candidate.exists():
                direction_file = candidate
                break
    if direction_file is None:
        print("No refusal_direction.npy found in C_weight_diff — C1 may not have run yet.")
        print("Run C1 first: python C1_weight_diff.py")
        sys.exit(1)
    label = "proxy_gemma_base"
else:
    if args.alllayer_dir:
        alllayer_dir = Path(args.alllayer_dir)
    else:
        alllayer_dir = DATA_ROOT / "paper_n250" / "google_gemma_2_9b_it"

    if args.direction_file:
        direction_file = Path(args.direction_file)
    else:
        # Auto-detect from C_weight_diff results
        cwdir = DATA_ROOT / "results" / "concept_evasion" / "C_weight_diff"
        direction_file = None
        if cwdir.exists():
            for pair_dir in sorted(cwdir.iterdir()):
                candidate = pair_dir / "refusal_direction.npy"
                if candidate.exists():
                    direction_file = candidate
                    break
        if direction_file is None:
            print("No refusal_direction.npy found. Run C1 first or pass --direction-file.")
            sys.exit(1)
    label = alllayer_dir.name

print(f"Direction: {direction_file}")
print(f"Alllayer : {alllayer_dir}")

# ---------------------------------------------------------------------------
# Load abliteration direction
# ---------------------------------------------------------------------------
d = np.load(direction_file).astype(np.float64)
d /= np.linalg.norm(d) + 1e-12
print(f"d shape: {d.shape}, norm: {np.linalg.norm(d):.6f}")

# ---------------------------------------------------------------------------
# Find available concepts
# ---------------------------------------------------------------------------
alllayer_paths = sorted(alllayer_dir.glob("calibration_alllayer_*.npy"))
concepts = [p.stem.replace("calibration_alllayer_", "") for p in alllayer_paths]
print(f"\nConcepts found: {len(concepts)} — {sorted(concepts)}")

if not concepts:
    print(f"No calibration_alllayer_*.npy files in {alllayer_dir}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Per-layer concept-direction cosine
# ---------------------------------------------------------------------------
# cos_matrix[concept][layer] = cosine(d, d_concept_L)
results: dict[str, dict] = {}

for concept, alllayer_path in zip(concepts, alllayer_paths):
    acts = np.load(alllayer_path).astype(np.float64)  # (n_layers, n_texts, hidden)
    n_layers, n_texts, hidden = acts.shape

    if hidden != d.shape[0]:
        print(f"  SKIP {concept}: hidden_dim mismatch ({hidden} vs {d.shape[0]})")
        continue

    # Split pos/neg: pos = first half, neg = second half
    n_pos = n_texts // 2
    pos_acts = acts[:, :n_pos, :]      # (n_layers, n_pos, hidden)
    neg_acts = acts[:, n_pos:, :]      # (n_layers, n_neg, hidden)

    per_layer_cos = []
    per_layer_dir_norm = []

    for L in range(n_layers):
        d_concept_L = pos_acts[L].mean(axis=0) - neg_acts[L].mean(axis=0)
        norm = np.linalg.norm(d_concept_L)
        per_layer_dir_norm.append(float(norm))
        if norm < 1e-9:
            per_layer_cos.append(0.0)
            continue
        d_concept_L /= norm
        cos = float(np.dot(d, d_concept_L))
        per_layer_cos.append(cos)

    abs_cos = [abs(c) for c in per_layer_cos]
    peak_L = int(np.argmax(abs_cos))
    peak_cos = per_layer_cos[peak_L]

    results[concept] = {
        "per_layer_cosine": per_layer_cos,
        "per_layer_dir_norm": per_layer_dir_norm,
        "peak_layer": peak_L,
        "peak_cosine": peak_cos,
        "mean_abs_cosine": float(np.mean(abs_cos)),
        "max_abs_cosine": float(max(abs_cos)),
    }
    print(f"  {concept:<20}  peak_cos={peak_cos:+.4f} @ L{peak_L:02d}  "
          f"mean|cos|={results[concept]['mean_abs_cosine']:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary = {
    "direction_file": str(direction_file),
    "alllayer_dir": str(alllayer_dir),
    "label": label,
    "n_layers": n_layers,
    "hidden_size": int(d.shape[0]),
    "concepts": {},
}

print(f"\n{'='*65}")
print(f"ABLITERATION DIRECTION FOOTPRINT — per-layer cosine with concept directions")
print(f"{'='*65}")
print(f"  {'Concept':<20} {'Peak L':>7}  {'cos @ peak':>10}  {'mean|cos|':>10}  {'max|cos|':>10}")
print("  " + "-" * 63)

rows = sorted(results.items(), key=lambda kv: -kv[1]["max_abs_cosine"])
for concept, v in rows:
    print(f"  {concept:<20} L{v['peak_layer']:<6d} {v['peak_cosine']:>+10.4f}  "
          f"{v['mean_abs_cosine']:>10.4f}  {v['max_abs_cosine']:>10.4f}")
    summary["concepts"][concept] = {
        "peak_layer": v["peak_layer"],
        "peak_cosine": v["peak_cosine"],
        "mean_abs_cosine": v["mean_abs_cosine"],
        "max_abs_cosine": v["max_abs_cosine"],
    }

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = RESULTS_DIR / "C_per_layer_cosine" / label
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "per_layer_cosine.json", "w") as f:
    json.dump(results, f, indent=2)

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved → {out_dir}")

# ---------------------------------------------------------------------------
# Optional heatmap
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concept_order = [c for c, _ in rows]
    matrix = np.array([results[c]["per_layer_cosine"] for c in concept_order])

    vmax = max(0.1, np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.28), max(4, len(concept_order) * 0.42)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(concept_order)))
    ax.set_yticklabels(concept_order, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_title(f"Abliteration direction d · concept direction (per layer)\n{label}")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    plt.tight_layout()
    heatmap_path = out_dir / "heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    print(f"Heatmap → {heatmap_path}")
except ImportError:
    print("(matplotlib not available — skipping heatmap)")

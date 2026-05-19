#!/usr/bin/env python3
"""
B2: Train CAZ-aware probes (depth-normalized, concept-specific layer placement).

For each concept: train probes at THREE positions from the CAZ profile:
  1. peak_layer — where separation is maximum (analogous to their layer 12)
  2. handoff_layer — onset of assembly (steepest velocity)
  3. ensemble of [handoff_layer, peak_layer, post_peak] — multi-layer monitor

The ensemble concatenates hidden states from all three layers before classifying.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR
from cea.data import BENIGN_CONCEPTS
from cea.extraction import load_model_and_tokenizer, extract_concept_reps
from cea.probes import train_linear_probe, evaluate_probe, save_probe

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--concepts", nargs="+", default=BENIGN_CONCEPTS)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--data-file", default=None)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
caz_dir = RESULTS_DIR / "B_caz" / model_slug
out_dir = RESULTS_DIR / "B_probes" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

caz_file = caz_dir / "caz_profiles.json"
if not caz_file.exists():
    print(f"CAZ profiles not found: {caz_file} — run B1 first")
    sys.exit(1)
with open(caz_file) as f:
    caz_profiles = json.load(f)

data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

model, tokenizer = load_model_and_tokenizer(args.model)
n_layers = model.config.num_hidden_layers + 1

results = {}
for concept in args.concepts:
    if concept not in caz_profiles or concept not in all_data:
        continue
    caz = caz_profiles[concept]
    peak_layer = caz["peak_layer"]
    handoff_layer = caz["handoff_layer"]
    zone_end = caz["assembly_zone"]["end"]
    # Post-peak: midpoint between peak and zone end (or peak+1 if zone_end == peak)
    post_peak = min(zone_end, min(peak_layer + max(1, (zone_end - peak_layer) // 2), n_layers - 1))

    caz_layers = sorted(set([handoff_layer, peak_layer, post_peak]))
    print(f"  {concept}: CAZ layers {caz_layers} "
          f"(handoff={handoff_layer}, peak={peak_layer}, post={post_peak})")

    concept_results = {}

    # 1. Single-layer probe at CAZ peak (their method but correct layer)
    reps_peak, labels = extract_concept_reps(
        all_data[concept], model, tokenizer,
        layer_indices=[peak_layer], batch_size=args.batch_size
    )
    h_peak = reps_peak[:, 0, :]
    n = len(labels)
    n_train = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, eval_idx = idx[:n_train], idx[n_train:]

    probe_peak = train_linear_probe(h_peak[train_idx], labels[train_idx])
    m_peak = evaluate_probe(probe_peak, h_peak[eval_idx], labels[eval_idx])
    save_probe(probe_peak, out_dir / f"{concept}_peak_layer{peak_layer}.pkl")
    concept_results["caz_peak"] = {"layers": [peak_layer], **m_peak}
    print(f"    [peak-only]    AUROC={m_peak['auroc']:.3f} | Recall@1%FPR={m_peak['recall_at_1pct_fpr']:.3f}")

    # 2. Single-layer probe at handoff boundary
    reps_handoff, _ = extract_concept_reps(
        all_data[concept], model, tokenizer,
        layer_indices=[handoff_layer], batch_size=args.batch_size
    )
    h_handoff = reps_handoff[:, 0, :]
    probe_handoff = train_linear_probe(h_handoff[train_idx], labels[train_idx])
    m_handoff = evaluate_probe(probe_handoff, h_handoff[eval_idx], labels[eval_idx])
    save_probe(probe_handoff, out_dir / f"{concept}_handoff_layer{handoff_layer}.pkl")
    concept_results["caz_handoff"] = {"layers": [handoff_layer], **m_handoff}
    print(f"    [handoff-only] AUROC={m_handoff['auroc']:.3f} | Recall@1%FPR={m_handoff['recall_at_1pct_fpr']:.3f}")

    # 3. Ensemble probe: concatenate hiddens at all CAZ layers
    reps_all, _ = extract_concept_reps(
        all_data[concept], model, tokenizer,
        layer_indices=caz_layers, batch_size=args.batch_size
    )
    h_ensemble = reps_all.reshape(len(labels), -1)  # (N, n_caz_layers * D)
    probe_ensemble = train_linear_probe(h_ensemble[train_idx], labels[train_idx])
    m_ensemble = evaluate_probe(probe_ensemble, h_ensemble[eval_idx], labels[eval_idx])
    save_probe(probe_ensemble, out_dir / f"{concept}_ensemble.pkl")
    concept_results["caz_ensemble"] = {"layers": caz_layers, **m_ensemble}
    print(f"    [ensemble]     AUROC={m_ensemble['auroc']:.3f} | Recall@1%FPR={m_ensemble['recall_at_1pct_fpr']:.3f}")

    results[concept] = concept_results

out_file = out_dir / "caz_probe_metrics.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved CAZ probe metrics → {out_file}")

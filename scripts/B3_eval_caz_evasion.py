#!/usr/bin/env python3
"""
B3: Compare evasion effectiveness on fixed-layer probes (Exp A) vs. CAZ-aware probes (Exp B).

Loads the fine-tuned evading model, extracts hidden states, applies both probe types,
and reports side-by-side recall drop.

Key question: Does the CAZ ensemble probe resist evasion better than the fixed-layer probe?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR
from cea.data import SAFETY_CONCEPTS, BENIGN_CONCEPTS
from cea.extraction import load_model_and_tokenizer, extract_concept_reps
from cea.probes import load_probe, evaluate_probe, train_linear_probe

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--probe-layer", type=int, default=None, help="Fixed probe layer (Exp A)")
parser.add_argument("--eval-concepts", nargs="+", default=SAFETY_CONCEPTS + BENIGN_CONCEPTS[:3])
parser.add_argument("--data-file", default=None)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
probe_dir_a = RESULTS_DIR / "A_probes" / model_slug
probe_dir_b = RESULTS_DIR / "B_probes" / model_slug
evasion_dir = RESULTS_DIR / "A_evasion" / model_slug
caz_dir = RESULTS_DIR / "B_caz" / model_slug
out_dir = RESULTS_DIR / "B_eval" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

caz_file = caz_dir / "caz_profiles.json"
with open(caz_file) as f:
    caz_profiles = json.load(f)

print("Loading clean model...")
clean_model, tokenizer = load_model_and_tokenizer(args.model)
n_layers = clean_model.config.num_hidden_layers + 1
fixed_probe_layer = args.probe_layer if args.probe_layer is not None else n_layers // 2

evasion_model_dir = evasion_dir / "model"
print(f"Loading fine-tuned model from {evasion_model_dir}...")
evading_model, _ = load_model_and_tokenizer(str(evasion_model_dir))

results = {}
for concept in args.eval_concepts:
    if concept not in all_data:
        continue
    print(f"\n  Evaluating: {concept}")

    # --- Fixed-layer probe (Experiment A) ---
    reps_clean_fixed, labels = extract_concept_reps(
        all_data[concept], clean_model, tokenizer,
        layer_indices=[fixed_probe_layer], batch_size=args.batch_size
    )
    h_clean_fixed = reps_clean_fixed[:, 0, :]
    probe_fixed = train_linear_probe(h_clean_fixed, labels)
    m_clean_fixed = evaluate_probe(probe_fixed, h_clean_fixed, labels)

    reps_evade_fixed, _ = extract_concept_reps(
        all_data[concept], evading_model, tokenizer,
        layer_indices=[fixed_probe_layer], batch_size=args.batch_size
    )
    h_evade_fixed = reps_evade_fixed[:, 0, :]
    m_evade_fixed = evaluate_probe(probe_fixed, h_evade_fixed, labels)
    drop_fixed = m_clean_fixed["recall_at_1pct_fpr"] - m_evade_fixed["recall_at_1pct_fpr"]

    # --- CAZ ensemble probe (Experiment B) ---
    # Get CAZ layers for this concept (or use fixed if no CAZ available)
    caz_layers_for_concept = None
    if concept in caz_profiles:
        caz = caz_profiles[concept]
        peak_layer = caz["peak_layer"]
        handoff_layer = caz["handoff_layer"]
        zone_end = caz["assembly_zone"]["end"]
        post_peak = min(zone_end, peak_layer + max(1, (zone_end - peak_layer) // 2))
        caz_layers_for_concept = sorted(set([handoff_layer, peak_layer, post_peak]))
    else:
        # Fallback: use fixed layer
        caz_layers_for_concept = [fixed_probe_layer]

    reps_clean_caz, _ = extract_concept_reps(
        all_data[concept], clean_model, tokenizer,
        layer_indices=caz_layers_for_concept, batch_size=args.batch_size
    )
    h_clean_caz = reps_clean_caz.reshape(len(labels), -1)
    probe_caz = train_linear_probe(h_clean_caz, labels)
    m_clean_caz = evaluate_probe(probe_caz, h_clean_caz, labels)

    reps_evade_caz, _ = extract_concept_reps(
        all_data[concept], evading_model, tokenizer,
        layer_indices=caz_layers_for_concept, batch_size=args.batch_size
    )
    h_evade_caz = reps_evade_caz.reshape(len(labels), -1)
    m_evade_caz = evaluate_probe(probe_caz, h_evade_caz, labels)
    drop_caz = m_clean_caz["recall_at_1pct_fpr"] - m_evade_caz["recall_at_1pct_fpr"]

    print(f"    Fixed (layer {fixed_probe_layer:2d}):   clean={m_clean_fixed['recall_at_1pct_fpr']:.3f} "
          f"evade={m_evade_fixed['recall_at_1pct_fpr']:.3f}  drop={drop_fixed:+.3f}")
    print(f"    CAZ   (layers {caz_layers_for_concept}): clean={m_clean_caz['recall_at_1pct_fpr']:.3f} "
          f"evade={m_evade_caz['recall_at_1pct_fpr']:.3f}  drop={drop_caz:+.3f}")
    print(f"    CAZ advantage (less drop): {drop_fixed - drop_caz:+.3f}")

    results[concept] = {
        "type": "safety" if concept in SAFETY_CONCEPTS else "benign",
        "fixed_probe": {
            "layer": fixed_probe_layer,
            "clean": m_clean_fixed,
            "evading": m_evade_fixed,
            "recall_drop": drop_fixed,
        },
        "caz_probe": {
            "layers": caz_layers_for_concept,
            "clean": m_clean_caz,
            "evading": m_evade_caz,
            "recall_drop": drop_caz,
        },
        "caz_advantage_recall_drop_reduction": drop_fixed - drop_caz,
    }

out_file = out_dir / "comparison.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved comparison → {out_file}")

print("\n=== FINAL COMPARISON ===")
print(f"{'Concept':15s} {'Type':6s} {'Fixed drop':>11s} {'CAZ drop':>10s} {'CAZ advantage':>14s}")
print("-" * 62)
for concept, r in results.items():
    print(f"{concept:15s} {r['type']:6s} "
          f"{r['fixed_probe']['recall_drop']:>+11.3f} "
          f"{r['caz_probe']['recall_drop']:>+10.3f} "
          f"{r['caz_advantage_recall_drop_reduction']:>+14.3f}")

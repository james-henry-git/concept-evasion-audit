#!/usr/bin/env python3
"""
A4: Evaluate evasion effectiveness on held-out safety probes.

Compares clean model vs. fine-tuned model recall@1%FPR on safety concepts.
This is the headline metric from the Neural Chameleons paper.
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
from cea.probes import train_linear_probe, evaluate_probe, save_probe

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--probe-layer", type=int, default=None)
parser.add_argument("--eval-concepts", nargs="+", default=SAFETY_CONCEPTS + BENIGN_CONCEPTS[:3])
parser.add_argument("--data-file", default=None)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
evasion_dir = RESULTS_DIR / "A_evasion" / model_slug
out_dir = RESULTS_DIR / "A_eval" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

# Load clean model
print("Loading clean model...")
clean_model, tokenizer = load_model_and_tokenizer(args.model)
n_layers = clean_model.config.num_hidden_layers + 1
probe_layer = args.probe_layer if args.probe_layer is not None else n_layers // 2

# Load fine-tuned model
evasion_model_dir = evasion_dir / "model"
print(f"Loading fine-tuned model from {evasion_model_dir}...")
evading_model, _ = load_model_and_tokenizer(str(evasion_model_dir))

results = {}
for concept in args.eval_concepts:
    if concept not in all_data:
        print(f"  Skipping {concept}: not in data")
        continue
    print(f"  Evaluating: {concept}")

    # Extract from clean model — train probe here on full data
    clean_reps, labels = extract_concept_reps(
        all_data[concept], clean_model, tokenizer,
        layer_indices=[probe_layer], batch_size=args.batch_size
    )
    h_clean = clean_reps[:, 0, :]
    probe = train_linear_probe(h_clean, labels)
    clean_metrics = evaluate_probe(probe, h_clean, labels)

    # Extract from evading model (same probe, their hidden states)
    evade_reps, _ = extract_concept_reps(
        all_data[concept], evading_model, tokenizer,
        layer_indices=[probe_layer], batch_size=args.batch_size
    )
    h_evade = evade_reps[:, 0, :]
    evade_metrics = evaluate_probe(probe, h_evade, labels)

    recall_drop = clean_metrics["recall_at_1pct_fpr"] - evade_metrics["recall_at_1pct_fpr"]
    print(f"    Clean recall@1%FPR: {clean_metrics['recall_at_1pct_fpr']:.3f}")
    print(f"    Evade recall@1%FPR: {evade_metrics['recall_at_1pct_fpr']:.3f}")
    print(f"    Drop: {recall_drop:.3f} ({'safety' if concept in SAFETY_CONCEPTS else 'benign'})")

    results[concept] = {
        "type": "safety" if concept in SAFETY_CONCEPTS else "benign",
        "probe_layer": probe_layer,
        "clean": clean_metrics,
        "evading": evade_metrics,
        "recall_drop_at_1pct_fpr": recall_drop,
    }

out_file = out_dir / "evasion_eval.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved evaluation results → {out_file}")

# Summary
print("\n=== SUMMARY ===")
for concept, r in results.items():
    print(f"  {concept:15s} [{r['type']:6s}]  drop={r['recall_drop_at_1pct_fpr']:+.3f}")

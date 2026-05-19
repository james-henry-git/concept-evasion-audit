#!/usr/bin/env python3
"""
A2: Train fixed-layer linear probes (Neural Chameleons methodology).

For each concept: extract hidden states at a single fixed layer, train a
logistic regression probe, save probe + metrics.
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
parser.add_argument("--probe-layer", type=int, default=None,
                    help="Fixed probe layer index (default: model_layers // 2)")
parser.add_argument("--concepts", nargs="+", default=BENIGN_CONCEPTS)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--data-file", default=None)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
out_dir = RESULTS_DIR / "A_probes" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

model, tokenizer = load_model_and_tokenizer(args.model)

n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
probe_layer = args.probe_layer if args.probe_layer is not None else n_layers // 2
print(f"Model: {args.model} | Layers: {n_layers} | Probe layer: {probe_layer}")

results = {}
for concept in args.concepts:
    if concept not in all_data:
        print(f"  Skipping {concept}: not in data")
        continue
    print(f"  Processing: {concept}")
    reps, labels = extract_concept_reps(
        all_data[concept], model, tokenizer,
        layer_indices=[probe_layer], batch_size=args.batch_size,
        desc=concept,
    )
    # reps: (N, 1, D) → (N, D)
    h = reps[:, 0, :]

    # Train/eval split (80/20)
    n = len(labels)
    n_train = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, eval_idx = idx[:n_train], idx[n_train:]

    probe = train_linear_probe(h[train_idx], labels[train_idx])
    metrics = evaluate_probe(probe, h[eval_idx], labels[eval_idx])
    print(f"    AUROC={metrics['auroc']:.3f} | Recall@1%FPR={metrics['recall_at_1pct_fpr']:.3f}")

    save_probe(probe, out_dir / f"{concept}_layer{probe_layer}.pkl")
    results[concept] = {"probe_layer": probe_layer, **metrics}

# Save results
out_file = out_dir / "probe_metrics.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved probe metrics → {out_file}")

#!/usr/bin/env python3
"""
A3: Fine-tune model for probe evasion (Neural Chameleons methodology).

Trains on benign concepts using dual KL + obfuscation loss.
Evaluates on safety probes afterward.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR
from cea.data import BENIGN_CONCEPTS
from cea.extraction import load_model_and_tokenizer
from cea.evasion import EvasionTrainer, build_lora_model
from cea.probes import load_probe

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--probe-layer", type=int, default=None)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--kl-weight", type=float, default=1.0)
parser.add_argument("--obf-weight", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--concepts", nargs="+", default=BENIGN_CONCEPTS)
parser.add_argument("--data-file", default=None)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
probe_dir = RESULTS_DIR / "A_probes" / model_slug
out_dir = RESULTS_DIR / "A_evasion" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

data_file = Path(args.data_file) if args.data_file else RESULTS_DIR / "concept_data.json"
with open(data_file) as f:
    all_data = json.load(f)

model, tokenizer = load_model_and_tokenizer(args.model)
n_layers = model.config.num_hidden_layers + 1
probe_layer = args.probe_layer if args.probe_layer is not None else n_layers // 2
print(f"Model: {args.model} | Probe layer: {probe_layer} | Epochs: {args.epochs}")

# Wrap with LoRA — only adapter weights are updated, base stays frozen.
# disable_adapter_layers() / enable_adapter_layers() give us the KL reference
# for free without a second copy of the model.
model = build_lora_model(model)
model.print_trainable_parameters()

# Load probes for training concepts
probes = {}
for concept in args.concepts:
    probe_path = probe_dir / f"{concept}_layer{probe_layer}.pkl"
    if probe_path.exists():
        probes[concept] = load_probe(probe_path)
    else:
        print(f"  Warning: no probe for {concept}, skipping")

if not probes:
    print("No probes found — run A2 first")
    sys.exit(1)

# Train evasion — one concept at a time per step
import random
concept_list = list(probes.keys())
all_logs = []

for epoch in range(args.epochs):
    random.shuffle(concept_list)
    epoch_losses = []
    for concept in concept_list:
        probe = probes[concept]
        trainer = EvasionTrainer(
            model, tokenizer, probe,
            probe_layer=probe_layer,
            kl_weight=args.kl_weight,
            obf_weight=args.obf_weight,
            lr=args.lr,
        )
        positives = all_data[concept]["positive"]
        step_log = trainer.train_step(concept, positives, batch_size=args.batch_size)
        epoch_losses.append(step_log)
        print(f"  Epoch {epoch+1} | {concept}: loss={step_log['loss']:.4f} "
              f"kl={step_log['kl_loss']:.4f} obf={step_log['obf_loss']:.4f}")
    all_logs.append(epoch_losses)

# Save fine-tuned model
evasion_model_dir = out_dir / "model"
trainer.save(evasion_model_dir)
print(f"\nSaved fine-tuned model → {evasion_model_dir}")

# Save training log
log_file = out_dir / "training_log.json"
with open(log_file, "w") as f:
    json.dump(all_logs, f, indent=2)
print(f"Saved training log → {log_file}")

#!/usr/bin/env python3
"""
C5: Extract behavioral refusal direction from RCP-style contrastive pairs.

Loads refusal pairs from a JSONL file, extracts last-token hidden states
from the specified model at every layer, and saves:

  behavioral_direction_L{peak}.npy   — normalised mean(pos) - mean(neg)
                                       at the peak layer (default L24)
  behavioral_direction_alllayer.npy  — (n_layers, hidden) at all layers
  extraction_meta.json               — provenance

This gives a clean d derived from behavioral signal rather than weight SVD,
with minimal contamination from non-refusal concepts.

Usage (GPU host):
    python C5_extract_behavioral_direction.py \\
        --model google/gemma-2-9b-it \\
        --pairs ~/Source/Rosetta_Chisel/pairs/refusal_pairs.jsonl \\
        --peak-layer 24
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="google/gemma-2-9b-it")
parser.add_argument("--pairs",      required=True,
                    help="Path to refusal_pairs.jsonl")
parser.add_argument("--peak-layer", type=int, default=24)
parser.add_argument("--n-pairs",    type=int, default=250,
                    help="Max pair_ids to use")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

model_slug = args.model.replace("/", "_")
out_dir = RESULTS_DIR / "C_behavioral_direction" / model_slug
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load pairs — one positive and one negative text per pair_id
#    Use first occurrence of each pair_id for each label.
# ------------------------------------------------------------------
print(f"Loading pairs from {args.pairs} ...")
pairs_path = Path(args.pairs).expanduser()

pos_texts, neg_texts = [], []
seen: dict[str, dict[int, str]] = {}  # pair_id -> {label: text}

with open(pairs_path) as f:
    for line in f:
        r = json.loads(line)
        pid = r["pair_id"]
        label = r["label"]
        if pid not in seen:
            seen[pid] = {}
        if label not in seen[pid]:
            seen[pid][label] = r["text"]

for pid, labels in list(seen.items())[:args.n_pairs]:
    if 0 in labels and 1 in labels:
        pos_texts.append(labels[1])
        neg_texts.append(labels[0])

print(f"  {len(pos_texts)} pairs loaded")
assert len(pos_texts) > 0, "No complete pairs found"

# ------------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading {args.model} → {device} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16,
    device_map=device, trust_remote_code=True
)
model.eval()
n_layers = model.config.num_hidden_layers
hidden   = model.config.hidden_size
print(f"  {n_layers} layers, hidden={hidden}")

# ------------------------------------------------------------------
# 3. Extract last-token hidden states at all layers
# ------------------------------------------------------------------
def extract_hidden(texts: list[str]) -> np.ndarray:
    """Returns (n_layers, n_texts, hidden) float32."""
    all_layers = [[] for _ in range(n_layers)]
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq, hidden)
        # skip index 0 (embedding layer), take layers 1..n_layers
        for L in range(n_layers):
            hs = out.hidden_states[L + 1]           # (batch, seq, hidden)
            last = hs[torch.arange(len(batch)),
                      enc["attention_mask"].sum(-1) - 1, :]  # (batch, hidden)
            all_layers[L].append(last.float().cpu().numpy())
        if (i // bs + 1) % 10 == 0:
            print(f"  batch {i//bs+1}/{(len(texts)+bs-1)//bs}")
    return np.stack([np.concatenate(all_layers[L], axis=0)
                     for L in range(n_layers)], axis=0)  # (n_layers, n, hidden)

print("\nExtracting positive activations ...")
pos_acts = extract_hidden(pos_texts)   # (n_layers, n, hidden)
print("Extracting negative activations ...")
neg_acts = extract_hidden(neg_texts)

# ------------------------------------------------------------------
# 4. Compute mean-difference direction at every layer
# ------------------------------------------------------------------
mean_diff = pos_acts.mean(axis=1) - neg_acts.mean(axis=1)  # (n_layers, hidden)
norms     = np.linalg.norm(mean_diff, axis=-1, keepdims=True)
directions = mean_diff / (norms + 1e-12)                    # (n_layers, hidden)

peak_dir  = directions[args.peak_layer].astype(np.float32)
peak_norm = float(norms[args.peak_layer])

print(f"\nPeak layer L{args.peak_layer}: direction norm (raw) = {peak_norm:.4f}")

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
np.save(out_dir / f"behavioral_direction_L{args.peak_layer}.npy",
        peak_dir)
np.save(out_dir / "behavioral_direction_alllayer.npy",
        directions.astype(np.float32))

meta = {
    "model": args.model,
    "pairs_file": str(pairs_path),
    "n_pairs": len(pos_texts),
    "peak_layer": args.peak_layer,
    "n_layers": n_layers,
    "hidden_size": hidden,
    "peak_raw_norm": peak_norm,
    "extracted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}
with open(out_dir / "extraction_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → {out_dir}")
print(f"  behavioral_direction_L{args.peak_layer}.npy  — shape {peak_dir.shape}")
print(f"  behavioral_direction_alllayer.npy  — shape {directions.shape}")
print(f"\nNext: python scripts/C2_transfer_abliteration.py \\")
print(f"      --skip-exp-a \\")
print(f"      --direction {out_dir}/behavioral_direction_L{args.peak_layer}.npy \\")
print(f"      --surgery-mode output_only")

#!/usr/bin/env python3
"""
C1: Weight diff analysis — extract the abliteration direction via SVD.

For each transformer layer, computes ΔW = W_abliterated - W_normal for the
matrices that write to the residual stream (o_proj, down_proj). The left
singular vector of ΔW gives the refusal direction in the residual stream.

Then cross-references with concept probe weight vectors to explain which
concepts share geometric space with the refusal direction, and why.

Outputs:
  C_weight_diff/{model_pair}/
    layer_svd.json       — singular values + rank check per layer per matrix
    refusal_direction.npy  — consensus refusal direction (hidden_size,)
    concept_cosine.json  — cosine sim: concept x layer x matrix_type
    summary.json         — damage prediction vs observed probe delta
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--normal-model",    default="google/gemma-2-9b-it")
parser.add_argument("--ablitd-model",    default="IlyaGusev/gemma-2-9b-it-abliterated")
parser.add_argument("--top-k",           type=int, default=8,
                    help="Number of singular values to extract per matrix")
parser.add_argument("--probes-dir",      default=None,
                    help="Override path to B_probes for normal model")
args = parser.parse_args()

normal_slug = args.normal_model.replace("/", "_")
ablitd_slug = args.ablitd_model.replace("/", "_")
pair_slug   = f"{normal_slug}_vs_{ablitd_slug}"

out_dir = RESULTS_DIR / "C_weight_diff" / pair_slug
out_dir.mkdir(parents=True, exist_ok=True)

probes_dir = Path(args.probes_dir) if args.probes_dir else \
    RESULTS_DIR / "B_probes" / normal_slug

# ------------------------------------------------------------------
# 1. Load concept probe weight vectors (peak-layer probes, normal model)
# ------------------------------------------------------------------
print("Loading concept probe vectors...")
concept_probes = {}   # concept -> {"layer": int, "vec": ndarray (hidden_size,)}
for pkl_path in sorted(probes_dir.glob("*_peak_layer*.pkl")):
    parts = pkl_path.stem.split("_peak_layer")
    concept = parts[0]
    layer   = int(parts[1])
    with open(pkl_path, "rb") as f:
        probe = pickle.load(f)
    vec = probe.coef_[0].astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    concept_probes[concept] = {"layer": layer, "vec": vec}

print(f"  Loaded {len(concept_probes)} concept probes: {sorted(concept_probes)}")

# ------------------------------------------------------------------
# 2. Load both models to CPU
# ------------------------------------------------------------------
from transformers import AutoModelForCausalLM

print(f"\nLoading {args.normal_model} to CPU...")
model_n = AutoModelForCausalLM.from_pretrained(
    args.normal_model, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)
model_n.eval()

print(f"Loading {args.ablitd_model} to CPU...")
model_a = AutoModelForCausalLM.from_pretrained(
    args.ablitd_model, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)
model_a.eval()

n_layers = model_n.config.num_hidden_layers
hidden   = model_n.config.hidden_size
print(f"  {n_layers} layers, hidden_size={hidden}")

# ------------------------------------------------------------------
# 3. Per-layer SVD of ΔW for o_proj and down_proj
# ------------------------------------------------------------------
# ΔW = W_ablit - W_normal  →  shape (out_dim, in_dim)
# Left singular vector (shape: out_dim = hidden_size) is the removed
# direction in the residual stream.

matrix_types = ["o_proj", "down_proj"]

layer_svd   = {}   # layer -> matrix_type -> {singular_values, direction, rank_ratio}
# Accumulate weighted refusal directions for consensus
direction_accum = np.zeros(hidden, dtype=np.float64)
weight_accum    = 0.0

print("\nProcessing layers...")
for i in range(n_layers):
    layer_n = model_n.model.layers[i]
    layer_a = model_a.model.layers[i]
    layer_svd[i] = {}

    for mtype in matrix_types:
        if mtype == "o_proj":
            W_n = layer_n.self_attn.o_proj.weight.float()   # (hidden, head_total)
            W_a = layer_a.self_attn.o_proj.weight.float()
        else:
            W_n = layer_n.mlp.down_proj.weight.float()      # (hidden, intermediate)
            W_a = layer_a.mlp.down_proj.weight.float()

        dW = W_a - W_n   # (hidden, in_dim)

        # Truncated SVD — only need top-k
        try:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        except Exception as e:
            print(f"  L{i} {mtype} SVD failed: {e}")
            continue

        k = min(args.top_k, len(S))
        sv  = S[:k].numpy().tolist()
        dir_vec = U[:, 0].numpy().astype(np.float64)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-12

        rank_ratio = float(sv[0] / sv[1]) if len(sv) > 1 and sv[1] > 1e-9 else float("inf")

        layer_svd[i][mtype] = {
            "singular_values": sv,
            "rank_ratio":      rank_ratio,
            "direction":       dir_vec.tolist(),
        }

        # Accumulate consensus direction, weighted by sv[0]
        direction_accum += dir_vec * sv[0]
        weight_accum    += sv[0]

    if (i + 1) % 10 == 0:
        print(f"  Layer {i+1}/{n_layers} done")

# ------------------------------------------------------------------
# 4. Consensus refusal direction
# ------------------------------------------------------------------
consensus_dir = direction_accum / (weight_accum + 1e-12)
consensus_dir /= np.linalg.norm(consensus_dir) + 1e-12

np.save(out_dir / "refusal_direction.npy", consensus_dir.astype(np.float32))
print(f"\nConsensus refusal direction saved.")

# ------------------------------------------------------------------
# 5. Cosine similarities: concept probe vs refusal direction per layer
# ------------------------------------------------------------------
print("\nComputing concept-direction cosine similarities...")
concept_cosine = {}

for concept, probe_info in concept_probes.items():
    probe_vec = probe_info["vec"]  # (hidden,) normalized
    peak_layer = probe_info["layer"]
    concept_cosine[concept] = {
        "peak_layer": peak_layer,
        "layers": {}
    }
    for i in range(n_layers):
        if i not in layer_svd:
            continue
        layer_entry = {}
        for mtype in matrix_types:
            if mtype not in layer_svd[i]:
                continue
            dir_vec = np.array(layer_svd[i][mtype]["direction"], dtype=np.float32)
            cos_sim = float(np.dot(probe_vec, dir_vec))
            layer_entry[mtype] = cos_sim
        # Average across matrix types
        vals = list(layer_entry.values())
        layer_entry["mean"] = float(np.mean(vals)) if vals else 0.0
        concept_cosine[concept]["layers"][i] = layer_entry

    # Highlight: cosine at concept's own peak layer
    if peak_layer in concept_cosine[concept]["layers"]:
        concept_cosine[concept]["cosine_at_peak"] = \
            concept_cosine[concept]["layers"][peak_layer]["mean"]

    # Cosine with consensus direction
    concept_cosine[concept]["cosine_consensus"] = \
        float(np.dot(probe_vec, consensus_dir.astype(np.float32)))

# ------------------------------------------------------------------
# 6. Summary: rank check + damage prediction
# ------------------------------------------------------------------
observed_probe_delta = {
    "deception": 0.0, "obfuscation": 0.0, "authorization": 0.0,
    "threat_severity": 0.0, "exfiltration": -0.050,
    "causation": 0.0, "certainty": 0.0, "credibility": 0.0,
    "negation": 0.0, "moral_valence": 0.0,
}

summary = {
    "model_pair": pair_slug,
    "n_layers": n_layers,
    "hidden_size": hidden,
    "rank_check": {},
    "concept_summary": {},
}

# Rank check: median rank_ratio per matrix type
for mtype in matrix_types:
    ratios = [layer_svd[i][mtype]["rank_ratio"]
              for i in range(n_layers) if mtype in layer_svd.get(i, {})]
    summary["rank_check"][mtype] = {
        "median_rank_ratio": float(np.median(ratios)),
        "min_rank_ratio":    float(np.min(ratios)),
        "max_rank_ratio":    float(np.max(ratios)),
        "pct_above_10x":     float(np.mean([r > 10 for r in ratios]) * 100),
    }

for concept in sorted(concept_cosine):
    cos_peak = concept_cosine[concept].get("cosine_at_peak", None)
    cos_cons = concept_cosine[concept].get("cosine_consensus", None)
    obs      = observed_probe_delta.get(concept, None)
    summary["concept_summary"][concept] = {
        "peak_layer":         concept_cosine[concept]["peak_layer"],
        "cosine_at_peak_layer": cos_peak,
        "cosine_consensus":   cos_cons,
        "observed_probe_delta": obs,
    }

# ------------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------------
# Strip direction vectors from layer_svd before saving JSON (large)
layer_svd_lean = {}
for i, mtypes in layer_svd.items():
    layer_svd_lean[i] = {}
    for mtype, entry in mtypes.items():
        layer_svd_lean[i][mtype] = {
            k: v for k, v in entry.items() if k != "direction"
        }

with open(out_dir / "layer_svd.json", "w") as f:
    json.dump(layer_svd_lean, f, indent=2)

with open(out_dir / "concept_cosine.json", "w") as f:
    json.dump(concept_cosine, f, indent=2)

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== RANK CHECK ===")
for mtype, rc in summary["rank_check"].items():
    print(f"  {mtype}: median rank_ratio={rc['median_rank_ratio']:.1f}x  "
          f"min={rc['min_rank_ratio']:.1f}x  {rc['pct_above_10x']:.0f}% layers >10x")

print(f"\n=== CONCEPT COSINE SIMILARITY AT PEAK LAYER ===")
print(f"  (alignment with abliteration direction; higher = more geometrically entangled)")
print(f"  {'Concept':<18} {'Peak L':>7}  {'|cos| at peak':>13}  {'|cos| consensus':>15}  {'Probe Δ':>8}")
print("  " + "-" * 70)
rows = [(abs(v["cosine_at_peak_layer"] or 0), c, v)
        for c, v in summary["concept_summary"].items()]
for _, concept, v in sorted(rows, reverse=True):
    cos_p = v["cosine_at_peak_layer"]
    cos_c = v["cosine_consensus"]
    obs   = v["observed_probe_delta"]
    print(f"  {concept:<18} L{v['peak_layer']:<6} "
          f"{abs(cos_p) if cos_p else 0:>13.4f}  "
          f"{abs(cos_c) if cos_c else 0:>15.4f}  "
          f"{obs if obs is not None else 'N/A':>8}")

print(f"\nSaved → {out_dir}")

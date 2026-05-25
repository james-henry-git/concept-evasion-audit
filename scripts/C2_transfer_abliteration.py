#!/usr/bin/env python3
"""
C2: PRH Transfer Abliteration + GEM-targeted surgery.

Two experiments in one script:

  Experiment A — PRH Transfer:
    Extract the refusal direction from C1 (Gemma weight diff), rotate it into
    Qwen2.5-7B-Instruct's representational space via Procrustes alignment on
    shared concept probe vectors, then apply abliteration surgery to Qwen.
    Tests whether PRH rotation generalises from concept alignment to causal
    intervention transfer.

  Experiment B — GEM-targeted surgery:
    Apply abliteration to Gemma (using d_svd from C1) but restricted to the
    layers where refusal actually assembles (the CAZ zone), rather than all
    layers. Predicts less collateral damage to adjacent concepts (exfiltration)
    because post-peak maintenance layers are left intact.

Surgery modes (--surgery-mode):
  full          — all layers, o_proj + down_proj  (classic abliteration)
  gem_targeted  — only layers within refusal CAZ zone
  peak_only     — only the single peak layer

Outputs saved to:
  C_transfer_abliteration/
    procrustes_rotation.npy     — R matrix (3584, 3584)
    procrustes_report.json      — anchor concepts, alignment quality
    qwen_abliterated_{mode}/    — modified Qwen weights
    gemma_gem_abliterated_{mode}/ — modified Gemma weights (Exp B)
    surgery_report.json         — which layers were modified, by how much
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--donor-model",    default="google/gemma-2-9b-it",
                    help="Model we extracted the refusal direction from")
parser.add_argument("--target-model",   default="Qwen/Qwen2.5-7B-Instruct",
                    help="Model to abliterate via PRH transfer")
parser.add_argument("--surgery-mode",   default="full",
                    choices=["full", "gem_targeted", "peak_only", "output_only"],
                    help="Which layers to modify")
parser.add_argument("--surgery-scale",  type=float, default=1.0,
                    help="Scale factor for the projection (1.0 = full removal)")
parser.add_argument("--run-exp-a",      action="store_true", default=True,
                    help="Run Exp A: PRH transfer to target model")
parser.add_argument("--run-exp-b",      action="store_true", default=True,
                    help="Run Exp B: GEM-targeted surgery on donor model")
parser.add_argument("--skip-exp-a",     action="store_true")
parser.add_argument("--skip-exp-b",     action="store_true")
parser.add_argument("--direction",      default=None,
                    help="Override refusal direction .npy (bypasses C1; use behavioral d from C5)")
parser.add_argument("--output-only-layers", type=int, default=5,
                    help="Number of final layers for output_only mode (default: 5)")
parser.add_argument("--exp-b-modes", nargs="+",
                    default=["gem_targeted", "peak_only", "output_only"],
                    choices=["gem_targeted", "peak_only", "output_only", "full"],
                    help="Which Exp B surgery modes to run (default: all three)")
parser.add_argument("--model-save-dir", default=None,
                    help="Override directory for saved surgery models (default: C_transfer_abliteration/)")
args = parser.parse_args()

run_exp_a = args.run_exp_a and not args.skip_exp_a
run_exp_b = args.run_exp_b and not args.skip_exp_b

donor_slug  = args.donor_model.replace("/", "_")
target_slug = args.target_model.replace("/", "_")

out_dir = RESULTS_DIR / "C_transfer_abliteration"
out_dir.mkdir(parents=True, exist_ok=True)

# Probe dirs
donor_probes_dir  = RESULTS_DIR / "B_probes" / donor_slug
target_probes_dir = RESULTS_DIR / "B_probes" / target_slug
c1_dir = RESULTS_DIR / "C_weight_diff" / f"{donor_slug}_vs_IlyaGusev_gemma-2-9b-it-abliterated"

# ------------------------------------------------------------------
# 1. Load C1 refusal direction (donor model space)
# ------------------------------------------------------------------
if args.direction:
    refusal_dir_path = Path(args.direction).expanduser()
    print(f"Using override direction: {refusal_dir_path}")
else:
    refusal_dir_path = c1_dir / "refusal_direction.npy"
    if not refusal_dir_path.exists():
        print(f"ERROR: C1 refusal direction not found at {refusal_dir_path}")
        print("Run C1_weight_diff.py first, or pass --direction <path>.")
        sys.exit(1)

d_refusal_donor = np.load(refusal_dir_path).astype(np.float64)
d_refusal_donor /= np.linalg.norm(d_refusal_donor) + 1e-12
print(f"Loaded refusal direction (norm={np.linalg.norm(d_refusal_donor):.4f})")

# Also load per-layer directions from C1 for GEM-targeted surgery
c1_layer_data = {}
layer_svd_path = c1_dir / "layer_svd.json"
if layer_svd_path.exists():
    with open(layer_svd_path) as f:
        raw = json.load(f)
    # Rekey as int
    for k, v in raw.items():
        c1_layer_data[int(k)] = v

# ------------------------------------------------------------------
# 2. Load CAZ profiles for refusal in donor model (for GEM targeting)
# ------------------------------------------------------------------
donor_refusal_caz_path = RESULTS_DIR / "B_caz" / donor_slug / "refusal_caz.json"
if donor_refusal_caz_path.exists():
    with open(donor_refusal_caz_path) as f:
        donor_refusal_caz = json.load(f)
    refusal_peak   = donor_refusal_caz["peak_layer"]
    refusal_zone_s = donor_refusal_caz["assembly_zone"]["start"]
    refusal_zone_e = donor_refusal_caz["assembly_zone"]["end"]
    refusal_depth  = donor_refusal_caz["peak_depth_normalized"]
    print(f"Donor refusal CAZ: peak=L{refusal_peak} ({refusal_depth:.0%} depth), "
          f"zone=L{refusal_zone_s}–{refusal_zone_e}")
else:
    # Fallback: use known Gemma values
    refusal_peak, refusal_zone_s, refusal_zone_e, refusal_depth = 24, 11, 42, 0.571
    print(f"No donor CAZ file — using defaults: peak=L{refusal_peak}, zone=L{refusal_zone_s}–{refusal_zone_e}")

# ------------------------------------------------------------------
# 3. Load concept probe vectors for Procrustes alignment
# ------------------------------------------------------------------
def load_probes(probes_dir):
    probes = {}
    for pkl_path in sorted(Path(probes_dir).glob("*_peak_layer*.pkl")):
        parts = pkl_path.stem.split("_peak_layer")
        concept = parts[0]
        with open(pkl_path, "rb") as f:
            probe = pickle.load(f)
        vec = probe.coef_[0].astype(np.float64)
        vec /= np.linalg.norm(vec) + 1e-12
        probes[concept] = vec
    return probes

print("\nLoading probe vectors...")
donor_probes  = load_probes(donor_probes_dir)
target_probes = load_probes(target_probes_dir) if run_exp_a else {}

shared_concepts = sorted(set(donor_probes) & set(target_probes))
print(f"  Donor concepts:  {sorted(donor_probes)}")
print(f"  Target concepts: {sorted(target_probes)}")
print(f"  Shared anchors:  {shared_concepts} ({len(shared_concepts)} concepts)")

# ------------------------------------------------------------------
# 4. Procrustes rotation: donor → target
# ------------------------------------------------------------------
procrustes_report = {}

if run_exp_a and len(shared_concepts) >= 3:
    hidden = d_refusal_donor.shape[0]
    A = np.stack([donor_probes[c]  for c in shared_concepts])  # (n, hidden)
    B = np.stack([target_probes[c] for c in shared_concepts])  # (n, hidden)

    # R = argmin ||B - A @ R||_F  s.t. R^T R = I
    # scipy orthogonal_procrustes: finds R s.t. A @ R ≈ B
    R, scale = orthogonal_procrustes(A, B)   # R: (hidden, hidden)

    # Alignment quality: mean cosine similarity of rotated anchors
    A_rot = A @ R   # (n, hidden)
    cos_sims = np.array([
        np.dot(A_rot[i], B[i]) / (np.linalg.norm(A_rot[i]) * np.linalg.norm(B[i]) + 1e-12)
        for i in range(len(shared_concepts))
    ])
    procrustes_report = {
        "anchor_concepts":   shared_concepts,
        "n_anchors":         len(shared_concepts),
        "hidden_dim":        hidden,
        "mean_cos_sim":      float(np.mean(cos_sims)),
        "per_concept":       {c: float(cs) for c, cs in zip(shared_concepts, cos_sims)},
        "note": (
            f"WARNING: {len(shared_concepts)} anchors in {hidden}-dim space — "
            "rotation is underdetermined; result is the minimum-norm solution. "
            "Reliable only in the subspace spanned by anchor concepts."
            if len(shared_concepts) < 20 else "OK"
        )
    }
    np.save(out_dir / "procrustes_rotation.npy", R.astype(np.float32))

    print(f"\nProcrustes rotation computed:")
    print(f"  Mean alignment cos: {np.mean(cos_sims):.4f}")
    for c, cs in zip(shared_concepts, cos_sims):
        print(f"    {c:<18} {cs:.4f}")
    print(f"  {procrustes_report['note']}")

    # Rotate refusal direction into target space
    d_refusal_target = R.T @ d_refusal_donor   # R maps donor→target, so R^T maps donor→target
    # (scipy orthogonal_procrustes: A @ R ≈ B, so R takes donor space to target space)
    d_refusal_target /= np.linalg.norm(d_refusal_target) + 1e-12

    with open(out_dir / "procrustes_report.json", "w") as f:
        json.dump(procrustes_report, f, indent=2)
else:
    d_refusal_target = None
    if run_exp_a:
        print(f"Not enough shared concepts ({len(shared_concepts)}) for rotation. Skipping Exp A.")
        run_exp_a = False

# ------------------------------------------------------------------
# Helper: apply abliteration surgery to a model
# ------------------------------------------------------------------
def abliterate_model(model, direction, layer_mask, mode_label, scale=1.0):
    """
    For each layer in layer_mask, project `direction` out of o_proj and down_proj.
    direction: (hidden,) unit vector in the model's residual stream space
    layer_mask: set of layer indices to modify
    Returns: dict of per-layer modification norms for reporting
    """
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / (d.norm() + 1e-12)
    d = d.to(next(model.parameters()).device)
    report = {}

    for i, layer in enumerate(model.model.layers):
        if i not in layer_mask:
            continue
        layer_report = {}
        for mtype, weight_obj in [
            ("o_proj",    layer.self_attn.o_proj.weight),
            ("down_proj", layer.mlp.down_proj.weight),
        ]:
            W = weight_obj.data.float()   # (hidden, in_dim)
            # Project out d from the output (row) space of W
            # W_new = W - scale * (W @ d.unsqueeze(1)) @ d.unsqueeze(0)  — wait
            # W writes to residual stream (dim 0 = hidden_size = direction space)
            # Project: W_new = W - scale * outer(W_col_proj, d)
            # Actually: W_new[row] -= scale * (d[row]) * (d^T @ W[row]) for each col
            # = W - scale * d.outer(d) @ W  ... no
            # Correct: remove the component of each OUTPUT vector in direction d
            # Each column of W is an "output direction" — project out d from each col
            # W_new = W - scale * d.unsqueeze(1) @ (d.unsqueeze(0) @ W)
            proj = scale * torch.outer(d, d @ W)   # (hidden, in_dim)
            delta_norm = float(proj.norm())
            weight_obj.data -= proj.to(weight_obj.dtype)
            layer_report[mtype] = {"delta_norm": delta_norm}

        report[i] = layer_report
        if (i + 1) % 10 == 0:
            print(f"  Modified through layer {i+1}")

    return report

# ------------------------------------------------------------------
# Helper: determine which layers to modify based on surgery mode
# ------------------------------------------------------------------
def get_layer_mask(mode, n_layers, peak_layer, zone_start, zone_end, peak_depth):
    if mode == "full":
        return set(range(n_layers))
    elif mode == "gem_targeted":
        return set(range(zone_start, zone_end + 1))
    elif mode == "peak_only":
        # ±2 layers around peak
        return set(range(max(0, peak_layer - 2), min(n_layers, peak_layer + 3)))
    elif mode == "output_only":
        # Final n layers before the output head — where the readout pathway lives
        n_out = getattr(args, "output_only_layers", 5)
        return set(range(max(0, n_layers - n_out), n_layers))
    return set(range(n_layers))

# ------------------------------------------------------------------
# 5. Experiment A: PRH transfer abliteration on target model
# ------------------------------------------------------------------
surgery_report = {}

if run_exp_a and d_refusal_target is not None:
    from transformers import AutoModelForCausalLM

    print(f"\n=== EXP A: PRH Transfer Abliteration → {args.target_model} ===")
    print(f"Surgery mode: {args.surgery_mode}")

    n_gpus = torch.cuda.device_count()
    device = "cuda:0" if n_gpus >= 1 else "cpu"

    print(f"Loading {args.target_model} → {device}...")
    model_target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    model_target.eval()

    n_layers_target = model_target.config.num_hidden_layers
    hidden_target   = model_target.config.hidden_size

    # Map donor refusal depth to target layer indices
    target_peak = int(round(refusal_depth * n_layers_target))
    # Proportionally map zone boundaries
    donor_n = 42  # donor n_layers (Gemma)
    target_zone_s = int(round(refusal_zone_s / donor_n * n_layers_target))
    target_zone_e = min(n_layers_target - 1,
                        int(round(refusal_zone_e / donor_n * n_layers_target)))

    print(f"  Target model: {n_layers_target} layers, hidden={hidden_target}")
    print(f"  Depth-mapped refusal zone: peak=L{target_peak}, "
          f"zone=L{target_zone_s}–{target_zone_e}")

    layer_mask = get_layer_mask(
        args.surgery_mode, n_layers_target,
        target_peak, target_zone_s, target_zone_e, refusal_depth
    )
    print(f"  Modifying {len(layer_mask)} layers: {sorted(layer_mask)[:5]}{'...' if len(layer_mask)>5 else ''}")

    mod_report = abliterate_model(
        model_target, d_refusal_target, layer_mask,
        args.surgery_mode, scale=args.surgery_scale
    )
    surgery_report["exp_a"] = {
        "target_model":   args.target_model,
        "surgery_mode":   args.surgery_mode,
        "layers_modified": sorted(layer_mask),
        "target_peak_layer": target_peak,
        "target_zone":    [target_zone_s, target_zone_e],
        "procrustes_mean_cos": procrustes_report.get("mean_cos_sim"),
        "per_layer": mod_report,
    }

    model_save_root = Path(args.model_save_dir) if args.model_save_dir else out_dir
    model_save_root.mkdir(parents=True, exist_ok=True)
    save_path = model_save_root / f"qwen_abliterated_{args.surgery_mode}"
    print(f"\n  Saving modified model → {save_path}")
    model_target.save_pretrained(save_path)
    del model_target
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 6. Experiment B: GEM-targeted surgery on donor model
# ------------------------------------------------------------------
if run_exp_b:
    from transformers import AutoModelForCausalLM

    print(f"\n=== EXP B: GEM-targeted surgery on {args.donor_model} ===")

    n_gpus = torch.cuda.device_count()
    device_b = "cuda:1" if n_gpus >= 2 else ("cuda:0" if n_gpus >= 1 else "cpu")

    print(f"Loading {args.donor_model} → {device_b}...")
    model_donor = AutoModelForCausalLM.from_pretrained(
        args.donor_model, torch_dtype=torch.bfloat16,
        device_map=device_b, trust_remote_code=True
    )
    model_donor.eval()

    n_layers_donor = model_donor.config.num_hidden_layers

    for mode in args.exp_b_modes:
        print(f"\n  Surgery mode: {mode}")
        layer_mask = get_layer_mask(
            mode, n_layers_donor,
            refusal_peak, refusal_zone_s, refusal_zone_e, refusal_depth
        )
        print(f"  Layers: {sorted(layer_mask)}")

        # Reload clean donor for each mode (don't compound modifications)
        if mode != "gem_targeted":
            del model_donor
            torch.cuda.empty_cache()
            print(f"  Reloading {args.donor_model}...")
            model_donor = AutoModelForCausalLM.from_pretrained(
                args.donor_model, torch_dtype=torch.bfloat16,
                device_map=device_b, trust_remote_code=True
            )
            model_donor.eval()

        mod_report = abliterate_model(
            model_donor, d_refusal_donor, layer_mask, mode,
            scale=args.surgery_scale
        )
        surgery_report[f"exp_b_{mode}"] = {
            "donor_model":     args.donor_model,
            "surgery_mode":    mode,
            "layers_modified": sorted(layer_mask),
            "peak_layer":      refusal_peak,
            "zone":            [refusal_zone_s, refusal_zone_e],
            "per_layer":       mod_report,
        }

        model_save_root = Path(args.model_save_dir) if args.model_save_dir else out_dir
        model_save_root.mkdir(parents=True, exist_ok=True)
        save_path = model_save_root / f"gemma_gem_abliterated_{mode}"
        print(f"  Saving → {save_path}")
        model_donor.save_pretrained(save_path)

    del model_donor
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 7. Save surgery report
# ------------------------------------------------------------------
with open(out_dir / "surgery_report.json", "w") as f:
    json.dump(surgery_report, f, indent=2)

print(f"\n=== DONE ===")
print(f"Outputs in {out_dir}/")
print(f"Next: run B1 (CAZ profiling) on saved models to measure scar pattern.")
print(f"  python scripts/B1_extract_caz.py --model {out_dir}/qwen_abliterated_{args.surgery_mode} --local")
print(f"  python scripts/B1_extract_caz.py --model {out_dir}/gemma_gem_abliterated_gem_targeted --local")

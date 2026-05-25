#!/usr/bin/env python3
"""
C4: RLHF + abliteration eigenvector comparison — per-layer concept direction drift.

For each concept, computes the mean-difference direction at every layer
(d_L = mean(pos[L]) - mean(neg[L]), normalised) for each model in a chain
and measures how those directions drift between models:

  base  →  instruct  →  abliterated

  cos(d_base_L, d_it_L)     — RLHF rotation (what instruction tuning did)
  cos(d_it_L,   d_ablit_L)  — abliteration damage (per-layer collateral)
  cos(d_base_L, d_ablit_L)  — total change

Also computes direction norms (signal strength) per layer per model —
RLHF centripetal hypothesis predicts stronger / earlier concept signals
in the instruct model.

Outputs (RESULTS_DIR/C_rlhf_comparison/):
  per_layer_alignment.json   — full concept × layer cosine matrix per model pair
  summary.json               — peak alignment / min alignment per concept per pair
  heatmap_{pair}.png         — heatmap per comparison pair (if matplotlib available)

Usage:
    # Available now (base + IT):
    python C4_rlhf_eigenvector_comparison.py

    # Full chain including abliterated (once alllayer extracted for that model):
    python C4_rlhf_eigenvector_comparison.py --include-abliterated
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
parser.add_argument("--include-abliterated", action="store_true",
                    help="Include abliterated model in comparison (requires alllayer extraction)")
parser.add_argument("--concepts", nargs="+", default=None,
                    help="Subset of concepts to analyse (default: all available)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model alllayer directories
# ---------------------------------------------------------------------------
DATA_ROOT = Path.home() / "rosetta_data"

MODELS = {
    "base":     DATA_ROOT / "paper_n250" / "google_gemma_2_9b",
    "instruct": DATA_ROOT / "paper_n250" / "google_gemma_2_9b_it",
}

if args.include_abliterated:
    ablit_dir = DATA_ROOT / "paper_n250" / "IlyaGusev_gemma-2-9b-it-abliterated"
    if not ablit_dir.exists():
        print(f"Abliterated alllayer dir not found: {ablit_dir}")
        print("Queue GPU extraction first: see C4 header for job command")
        sys.exit(1)
    MODELS["abliterated"] = ablit_dir

# Check availability
for name, d in MODELS.items():
    count = len(list(d.glob("calibration_alllayer_*.npy")))
    print(f"{name}: {count} alllayer files in {d.name}")

PAIRS = [("base", "instruct")]
if "abliterated" in MODELS:
    PAIRS += [("instruct", "abliterated"), ("base", "abliterated")]

# ---------------------------------------------------------------------------
# Load alllayer activations for all models and concepts
# ---------------------------------------------------------------------------
# Find concepts present in ALL models
concept_sets = {}
for name, d in MODELS.items():
    concept_sets[name] = {
        p.stem.replace("calibration_alllayer_", "")
        for p in d.glob("calibration_alllayer_*.npy")
    }

shared_concepts = set.intersection(*concept_sets.values())
if args.concepts:
    shared_concepts = shared_concepts & set(args.concepts)
shared_concepts = sorted(shared_concepts)
print(f"\nConcepts available in all models: {len(shared_concepts)} — {shared_concepts}")


def load_per_layer_directions(model_dir: Path, concepts: list[str]) -> dict[str, np.ndarray]:
    """Returns {concept: ndarray of shape (n_layers, hidden)} — normalised directions."""
    result = {}
    for concept in concepts:
        path = model_dir / f"calibration_alllayer_{concept}.npy"
        acts = np.load(path).astype(np.float64)  # (n_layers, n_texts, hidden)
        n_layers, n_texts, hidden = acts.shape
        n_pos = n_texts // 2
        pos_acts = acts[:, :n_pos, :]
        neg_acts = acts[:, n_pos:, :]
        dirs = pos_acts.mean(axis=1) - neg_acts.mean(axis=1)  # (n_layers, hidden)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = dirs / (norms + 1e-12)
        result[concept] = dirs
    return result


print("\nLoading activations...")
model_dirs: dict[str, dict[str, np.ndarray]] = {}
for name, d in MODELS.items():
    model_dirs[name] = load_per_layer_directions(d, shared_concepts)
    print(f"  {name}: loaded {len(model_dirs[name])} concepts")

n_layers = next(iter(next(iter(model_dirs.values())).values())).shape[0]
print(f"  n_layers = {n_layers}")

# ---------------------------------------------------------------------------
# Per-pair, per-concept, per-layer cosine
# ---------------------------------------------------------------------------
all_results = {}

for (model_a, model_b) in PAIRS:
    pair_key = f"{model_a}_vs_{model_b}"
    print(f"\n{pair_key}")
    pair_result = {}

    for concept in shared_concepts:
        da = model_dirs[model_a][concept]   # (n_layers, hidden)
        db = model_dirs[model_b][concept]

        # Per-layer cosine: sum(da[L] * db[L]) since both are unit vectors
        per_layer_cos = (da * db).sum(axis=-1).tolist()  # (n_layers,)

        # Also: direction norm ratio (how much signal does RLHF add/remove?)
        # We need the raw unnormalised directions for this, so reload briefly
        path_a = MODELS[model_a] / f"calibration_alllayer_{concept}.npy"
        path_b = MODELS[model_b] / f"calibration_alllayer_{concept}.npy"
        acts_a = np.load(path_a).astype(np.float64)
        acts_b = np.load(path_b).astype(np.float64)
        np_a = acts_a.shape[1] // 2
        np_b = acts_b.shape[1] // 2
        raw_a = (acts_a[:, :np_a, :].mean(1) - acts_a[:, np_a:, :].mean(1))
        raw_b = (acts_b[:, :np_b, :].mean(1) - acts_b[:, np_b:, :].mean(1))
        norm_a = np.linalg.norm(raw_a, axis=-1)
        norm_b = np.linalg.norm(raw_b, axis=-1)
        # ratio > 1 means b has stronger signal
        norm_ratio = (norm_b / (norm_a + 1e-9)).tolist()

        abs_cos = [abs(c) for c in per_layer_cos]
        min_cos_layer = int(np.argmin(abs_cos))
        peak_cos_layer = int(np.argmax(abs_cos))

        pair_result[concept] = {
            "per_layer_cosine": per_layer_cos,
            "per_layer_norm_ratio_b_over_a": norm_ratio,
            "min_cosine_layer": min_cos_layer,
            "min_cosine": float(min(per_layer_cos)),
            "max_abs_cosine_layer": peak_cos_layer,
            "max_abs_cosine": float(per_layer_cos[peak_cos_layer]),
            "mean_cosine": float(np.mean(per_layer_cos)),
        }

        print(f"  {concept:<22} mean_cos={pair_result[concept]['mean_cosine']:+.4f}  "
              f"min_cos={pair_result[concept]['min_cosine']:+.4f} @ L{min_cos_layer:02d}")

    all_results[pair_key] = pair_result

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary = {
    "models": list(MODELS.keys()),
    "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
    "n_layers": n_layers,
    "concepts": shared_concepts,
    "findings": {},
}

for pair_key, pair_result in all_results.items():
    summary["findings"][pair_key] = {}
    for concept, v in sorted(pair_result.items(),
                              key=lambda kv: kv[1]["mean_cosine"]):
        summary["findings"][pair_key][concept] = {
            "mean_cosine": v["mean_cosine"],
            "min_cosine": v["min_cosine"],
            "min_cosine_layer": v["min_cosine_layer"],
        }

print(f"\n{'='*68}")
print("RLHF EIGENVECTOR DRIFT SUMMARY")
print(f"{'='*68}")
for pair_key, pair_result in all_results.items():
    model_a, model_b = pair_key.split("_vs_")
    print(f"\n  {model_a} → {model_b}")
    print(f"  {'Concept':<22} {'mean cos':>9}  {'min cos':>9}  {'min @ L':>8}")
    print("  " + "-" * 55)
    rows = sorted(pair_result.items(), key=lambda kv: kv[1]["mean_cosine"])
    for concept, v in rows:
        print(f"  {concept:<22} {v['mean_cosine']:>+9.4f}  "
              f"{v['min_cosine']:>+9.4f}  L{v['min_cosine_layer']:<6d}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = RESULTS_DIR / "C_rlhf_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "per_layer_alignment.json", "w") as f:
    json.dump(all_results, f, indent=2)

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved → {out_dir}")

# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pair_key, pair_result in all_results.items():
        concept_order = sorted(pair_result.keys(),
                               key=lambda c: pair_result[c]["mean_cosine"])
        matrix = np.array([pair_result[c]["per_layer_cosine"] for c in concept_order])

        fig, axes = plt.subplots(2, 1, figsize=(max(12, n_layers * 0.3),
                                                max(7, len(concept_order) * 0.6)),
                                 gridspec_kw={"height_ratios": [3, 1]})

        # Cosine heatmap
        im = axes[0].imshow(matrix, aspect="auto", cmap="RdBu_r",
                            vmin=-1, vmax=1, interpolation="nearest")
        axes[0].set_yticks(range(len(concept_order)))
        axes[0].set_yticklabels(concept_order, fontsize=9)
        axes[0].set_xlabel("Layer")
        model_a, model_b = pair_key.split("_vs_")
        axes[0].set_title(
            f"cos(d_concept_L[{model_a}], d_concept_L[{model_b}])\n"
            f"1.0 = same direction, 0 = orthogonal, −1 = flipped"
        )
        plt.colorbar(im, ax=axes[0], label="cosine alignment")

        # Mean cosine profile across all concepts
        mean_cos = matrix.mean(axis=0)
        axes[1].plot(range(n_layers), mean_cos, color="steelblue", linewidth=1.5)
        axes[1].axhline(0, color="grey", linewidth=0.5, linestyle="--")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Mean cosine")
        axes[1].set_title("Mean alignment across all concepts")
        axes[1].set_ylim(-1.05, 1.05)

        plt.tight_layout()
        path = out_dir / f"heatmap_{pair_key}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Heatmap → {path}")

except ImportError:
    print("(matplotlib not available — skipping heatmaps)")

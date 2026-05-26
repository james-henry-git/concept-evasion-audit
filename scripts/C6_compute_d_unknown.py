#!/usr/bin/env python3
"""
C6: Compute d_unknown_L24 on GPU host.

Projects IlyaGusev's refusal direction onto the span of known concept directions
at layer L24 and saves the orthogonal residual (d_unknown) — the component of
his abliteration direction that is NOT explained by labeled concepts.

Requires:
  ~/rosetta_data/paper_n250/google_gemma_2_9b_it/calibration_alllayer_*.npy
  ~/rosetta_data/results/concept_evasion/C_weight_diff/.../refusal_direction.npy
"""
import numpy as np
from pathlib import Path

PEAK_LAYER   = 24
ALLLAYER_DIR = Path.home() / "rosetta_data/paper_n250/google_gemma_2_9b_it"
WEIGHT_DIR   = Path.home() / (
    "rosetta_data/results/concept_evasion/C_weight_diff/"
    "google_gemma-2-9b-it_vs_IlyaGusev_gemma-2-9b-it-abliterated"
)

d_ilya = np.load(WEIGHT_DIR / "refusal_direction.npy").astype(np.float64)
d_ilya /= np.linalg.norm(d_ilya) + 1e-12

paths = sorted(ALLLAYER_DIR.glob("calibration_alllayer_*.npy"))
print(f"Loading {len(paths)} concept alllayer files at L{PEAK_LAYER} ...")

dirs = []
for p in paths:
    acts = np.load(p).astype(np.float64)   # (n_layers, n_texts, hidden)
    n = acts.shape[1] // 2
    d_c = acts[PEAK_LAYER, :n, :].mean(0) - acts[PEAK_LAYER, n:, :].mean(0)
    norm = np.linalg.norm(d_c)
    if norm > 1e-9:
        dirs.append(d_c / norm)

C  = np.stack(dirs)                              # (n_concepts, hidden)
Cd = C @ d_ilya
CC = C @ C.T + 1e-6 * np.eye(len(dirs))
coeffs      = np.linalg.solve(CC, Cd)
d_known_raw = C.T @ coeffs
d_unknown_raw = d_ilya - d_known_raw
d_unknown = (d_unknown_raw / (np.linalg.norm(d_unknown_raw) + 1e-12)).astype(np.float32)

R2 = float(np.linalg.norm(d_known_raw) ** 2)
print(f"R2 explained by known concepts: {R2:.4f}  ({R2*100:.2f}%)")
print(f"d_unknown norm (should be 1.0): {np.linalg.norm(d_unknown):.6f}")

out = WEIGHT_DIR / f"d_unknown_L{PEAK_LAYER}.npy"
np.save(out, d_unknown)
print(f"Saved → {out}  shape={d_unknown.shape}")

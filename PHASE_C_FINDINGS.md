# Phase C Findings — d_unknown Surgery and Refusal Direction Decomposition

*Written: 2026-05-26 07:03 UTC*

## Setup

**Research question**: IlyaGusev's abliteration direction (`d_ilya`, extracted via weight-diff SVD) is
nearly orthogonal to the behaviorally-extracted refusal direction (`d_behav`, from C5 contrastive
last-token probing). If `d_ilya` doesn't point at the labeled concept space, what *is* it removing?
Can we decompose it and test whether the unexplained residual is the active ingredient?

**Model**: `google/gemma-2-9b-it` (42 layers, D=3584)  
**Comparison**: `IlyaGusev/gemma-2-9b-it-abliterated` (published weight-diff baseline)  
**Peak layer for decomposition**: L24 (determined by CAZ profile — highest Fisher separation
for refusal concept)

---

## C6: Direction Decomposition

Using 17 labeled Rosetta concept directions at L24, we project `d_ilya` onto their span and
extract the orthogonal residual `d_unknown`.

| Quantity | Value |
|----------|-------|
| R² explained by 17 labeled concepts | 6.07% |
| Residual unexplained (`d_unknown` as fraction of `d_ilya`) | 93.93% |
| cos(`d_ilya`, `d_unknown`) | 0.9692 |
| cos(`d_ilya`, `d_known`) | 0.2464 |
| cos(`d_ilya`, `d_behav`) | −0.13 |

**`d_unknown ≈ d_ilya`**: with only 6% explained by the entire labeled concept vocabulary,
`d_unknown` is almost the same vector as `d_ilya`. IlyaGusev's abliteration direction is
nearly orthogonal to all 17 labeled semantic concepts.

**`d_behav ⊥ d_ilya`**: the behaviorally-extracted refusal direction (C5 contrastive last-token)
is near-orthogonal to IlyaGusev's weight-diff direction (cos = −0.13). These are two distinct
objects. `d_behav` captures something the probe sees in the *activation space*; `d_ilya` is
what IlyaGusev actually removed from the *weight space*. Prior Exp B surgery used `d_behav` —
that was not testing IlyaGusev's method.

Per-concept cos² contributions (all very small, confirming no single labeled concept drives the direction):

| Concept | cos² |
|---------|------|
| sentiment | 0.024 |
| moral_valence | 0.015 |
| authorization | 0.007 |
| exfiltration | 0.007 |
| credibility | 0.005 |
| negation | 0.004 |
| deception | 0.004 |
| causation | 0.003 |
| ... | < 0.003 each |

---

## Surgery Conditions

Two surgery conditions applied to `google/gemma-2-9b-it` using `d_unknown` (scale=1.0):

- **`peak_only`**: `W_new = W − outer(d, d @ W)` applied to `o_proj` + `down_proj` at L24 only
- **`gem_targeted`**: same formula applied at all GEM assembly-event layers (plus L24)

Surgery formula removes `d` from the output (row) space of each weight matrix. Post-surgery,
no column of W can project in direction `d` — the concept direction is erased from the
model's writable subspace at those layers.

---

## Concept Retention Results

Fisher separation S = ‖μ₊ − μ₋‖² / (½σ²₊ + ½σ²₋). Higher = more separable.
Retention = S_surgery / S_baseline.

### Overlapping benign concepts (5 shared between baseline and surgery runs)

| Concept | Baseline S | IlyaGusev ret% | peak_only ret% | gem_targeted ret% |
|---------|-----------|----------------|----------------|-------------------|
| causation | 1942.1 | 98.96% | **99.81%** | **99.07%** |
| certainty | 2691.7 | 99.69% | **100.00%** | **99.93%** |
| credibility | 6195.2 | 99.98% | **100.00%** | **100.00%** |
| moral_valence | 1392.0 | 93.93% | **99.97%** | **97.34%** |
| negation | 2046.6 | 100.03% | **100.00%** | **100.00%** |

**`d_unknown` surgery is more concept-preserving than IlyaGusev's full abliteration** on all
five tested concepts. `peak_only` retains 99.8–100%; `gem_targeted` retains 97–100%.
IlyaGusev's published model drops to 91–94% on `exfiltration` and `threat_severity`
(not tested in surgery runs, but included in the IlyaGusev comparison above).

### Refusal concept (diagnostic, 250 pairs)

| Condition | S_peak (L24) |
|-----------|-------------|
| peak_only | 14,353 |
| gem_targeted | 13,425 |

Refusal separation remains very high under both surgery conditions. The `d_unknown` direction
is *not* the primary driver of refusal geometry — the concept assembles robustly even after
its removal from L24.

### IlyaGusev full-model retention for reference

| Concept | Baseline S | IlyaGusev S | Retention |
|---------|-----------|-------------|-----------|
| exfiltration | 1496.6 | 1374.3 | 91.83% |
| threat_severity | 2670.5 | 2440.7 | 91.39% |
| authorization | 2643.6 | 2502.4 | 94.66% |
| moral_valence | 1392.0 | 1307.4 | 93.93% |
| deception | 856.4 | 846.3 | 98.82% |
| causation | 1942.1 | 1921.9 | 98.96% |
| certainty | 2691.7 | 2683.4 | 99.69% |
| credibility | 6195.2 | 6194.2 | 99.98% |
| obfuscation | 2058.4 | 2049.4 | 99.56% |
| negation | 2046.6 | 2047.3 | 100.03% |

The collateral damage in IlyaGusev's model likely comes from his surgery operating across all
layers and all projection types, not just o_proj/down_proj at peak layer.

---

## Infrastructure Fixes (this phase)

Three bugs were fixed in `concept-evasion-audit` during this phase. All commits are in the repo.

### 1. B1 `--out-dir` argument missing (commit 8269c23)
`B1_extract_caz.py` had no way to write results to an alternate path. When `/home/coder` on the
GPU host was at 100% capacity, jobs failed. Added `--out-dir` argument; GPU jobs now write to
`/tmp/` first, then `mv` results to the canonical location.

### 2. Multi-GPU device mismatch in surgery (commit 4d88fc6)
`abliterate_model` in `C2_transfer_abliteration.py` moved the direction vector `d` to
`next(model.parameters()).device` (cuda:0), but with `device_map="auto"` layers 22+ were on
cuda:1. The outer-product `W - outer(d, d @ W)` then failed with a cross-device tensor error.
Fix: move `d` to `W.device` inside the per-weight loop, so each surgery operation is
device-local.

### 3. Tokenizer save omission (commit 637dcbc)
`model.save_pretrained(path)` saves model weights and config but **not the tokenizer**.
It also sets `config._name_or_path` to the local save path. `AutoTokenizer.from_pretrained(local_path)` then finds no tokenizer files and can't resolve the original model name → returns a broken/default tokenizer → all texts tokenize identically → μ₊ = μ₋ → Fisher separation = 0.0 at every layer, including the embedding layer.

This manifested as: all B_caz separations = 0.0 (a diagnostic red flag — even the embedding
layer should have non-zero separation for any real concept). Fix: add
`tokenizer.save_pretrained(save_path)` to C2 after each `model.save_pretrained`.

**Diagnostic signal**: embedding-layer separation = 0 indicates a tokenizer failure, not a
model geometry result. If `S(L0) = 0`, the tokenizer is broken.

### 4. Refusal diagnostic pairs expanded (commit 81734c5)
`diagnostic_concepts.py` had 25 hardcoded refusal pairs (fallback). Added `_refusal_pairs_250.py`
auto-generated from `~/Source/Rosetta_Chisel/pairs/refusal_pairs.jsonl` (250 pairs: label=1
→ positive, label=0 → negative). The module is imported at startup; if absent on GPU hosts,
the 25-pair fallback is used. This brought refusal extraction from ~7 batches to ~32 batches,
giving much more reliable Fisher separation estimates.

---

## Interpretation

1. **d_ilya is not a semantic concept direction.** It explains only 6% of its own variance via
   17 labeled concepts spanning the Rosetta vocabulary. IlyaGusev is removing something that
   has no name in our current concept ontology.

2. **d_unknown surgery is nearly lossless.** Surgical removal of d_unknown at L24 leaves benign
   concept geometry 99–100% intact. This is *more* conservative than IlyaGusev's full model
   (which drops to ~91% on some concepts). The surgery formula is working correctly: it removes
   only the d direction from the row space without disturbing orthogonal directions.

3. **Refusal geometry survives d_unknown removal.** The refusal concept (S ≈ 14,000) is robust
   to both peak_only and gem_targeted surgery. If IlyaGusev's direction abliterates refusal
   *behavior*, it is not doing so by destroying the refusal *representation*. The mechanism
   is likely routing disruption, not representation erasure.

4. **d_behav and d_ilya are orthogonal objects** (cos = −0.13). Future work should probe
   d_behav surgery separately to test whether behavioral refusal extraction yields a different
   selectivity profile.

---

## Data Artifacts

All results uploaded to HF `james-ra-henry/Rosetta-Activations`:

- `results/concept_evasion/B_caz/gemma_d_unknown_peak_only/`
- `results/concept_evasion/B_caz/gemma_d_unknown_gem_targeted/`
- `results/concept_evasion/C_weight_diff/google_gemma-2-9b-it_vs_IlyaGusev_gemma-2-9b-it-abliterated/`
  (includes `d_unknown_L24.npy`, `direction_decomposition_L24.json`, `d_unknown_alllayer_summary.json`)

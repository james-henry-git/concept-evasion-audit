# Concept Evasion Audit

*Written: 2026-05-19 UTC | Updated: 2026-05-26 07:03 UTC*

A methodology challenge and replication study of:

> **Neural Chameleons: Language Models Learning to Hide Their Thoughts** (arXiv 2512.11949)

## The Question

Neural Chameleons trains linear probes at a **fixed layer** to detect concepts, then shows models
can be fine-tuned to evade those probes. The probes generalize across architectures, and evasion
transfers zero-shot to unseen safety-relevant concepts.

Their setup rests on two assumptions:
1. Concepts live at a fixed layer — layer 12 for Gemma-2-9b-it (CAZ shows this is false)
2. Their abliterated base model is a valid probe-training substrate (we test this directly)

**Does their finding replicate? And if it does, is it because of their methodology or despite it?**

---

## Status

| Phase | Question | Verdict |
|-------|----------|---------|
| **A** — Baseline replication | Does the evasion attack work on an intact model? | **No** — zero probe drop, LoRA stuck |
| **B** — CAZ-aware probing | Does abliteration damage concept geometry? | **Surgical only** — 9/10 zero-delta |
| **C** — Direction decomposition | What direction does IlyaGusev actually remove? | **Not semantic** — 6% explained by 17 concepts |

**Bottom line**: The evasion attack is specific to their abliterated-model setup. On intact safety-tuned
models, concept representations resist LoRA fine-tuning completely. The abliteration direction is
nearly orthogonal to all labeled semantic concepts, and surgically removing it leaves benign concept
geometry 99–100% intact.

---

## Experiments

### Phase A — Baseline Replication

Reproduce Neural Chameleons' probe + evasion pipeline on a standard safety-tuned model.

- `A1_generate_data.py` — concept pair data from Rosetta Concept Pairs v1
- `A2_train_probes.py` — fixed-layer logistic regression probes
- `A3_train_evasion.py` — LoRA evasion fine-tuning (dual KL + obfuscation loss)
- `A4_eval_evasion.py` — clean-text probe recall after evasion
- `A5_eval_triggered.py` — evaluation with trigger prefix

**Model**: `Qwen/Qwen2.5-7B-Instruct`  
**Finding**: Zero evasion. Obfuscation loss stuck at ~0.99 throughout training. See `PHASE_A_FINDINGS.md`.

### Phase B — CAZ-Aware Probing

Test whether depth-normalized, assembly-zone-targeted probes outperform fixed-layer probes,
and whether IlyaGusev's abliteration damages safety concept representations.

- `B1_extract_caz.py` — CAZ profiles (separation, coherence, velocity) per concept per layer
- `B2_train_caz_probes.py` — probes at peak / handoff / ensemble positions
- `B3_eval_caz_evasion.py` — evasion evaluation (not run — requires Neural Chameleons' data)

**Models**: `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-9b-it`, `IlyaGusev/gemma-2-9b-it-abliterated`  
**Finding**: CAZ probes ≥ fixed-layer; abliteration leaves 9/10 concepts at zero-delta. See `PHASE_B_FINDINGS.md`.

### Phase C — Direction Decomposition and Surgical Abliteration

Characterise IlyaGusev's actual abliteration direction (`d_ilya`) by decomposing it against
the labeled concept vocabulary, then test surgical removal of the unexplained residual.

- `C1_weight_diff.py` — SVD of weight diffs → `d_ilya`, trench-depth profile across layers
- `C2_transfer_abliteration.py` — apply `W ← W − outer(d, d@W)` surgery to any direction
- `C3_per_layer_concept_cosine.py` — cos(`d_ilya`, concept directions) across all layers
- `C4_rlhf_eigenvector_comparison.py` — compare abliteration direction to RLHF alignment geometry
- `C5_extract_behavioral_direction.py` — contrastive last-token refusal direction (`d_behav`)
- `C6_compute_d_unknown.py` — project `d_ilya` onto concept span; extract orthogonal residual `d_unknown`

**Model**: `google/gemma-2-9b-it` (42 layers, D=3584), decomposition at L24 (CAZ peak for refusal)  
**Finding**: `d_ilya` is 94% unexplained by 17 labeled concepts; surgical removal of `d_unknown`
retains benign concept geometry 99–100%. See `PHASE_C_FINDINGS.md`.

---

## Key Results Summary

### Phase A

On `Qwen2.5-7B-Instruct` (intact safety training):
- Probe accuracy: AUROC=1.000, Recall@1%FPR=1.000 across all 13 concepts
- Evasion drop after LoRA fine-tuning: **0.000** universally
- Trigger prefix effect: noise-level (±0.002)

### Phase B — Abliteration damage to concept geometry

Comparing `google/gemma-2-9b-it` vs `IlyaGusev/gemma-2-9b-it-abliterated`:

| Concept | Normal R@1% | Ablitd R@1% | Δ |
|---------|------------|-------------|---|
| authorization | 1.000 | 1.000 | 0.000 |
| deception | 0.980 | 0.980 | 0.000 |
| **exfiltration** | 0.960 | **0.911** | **−0.050** |
| obfuscation | 1.000 | 1.000 | 0.000 |
| threat_severity | 1.000 | 1.000 | 0.000 |
| causation | 1.000 | 1.000 | 0.000 |
| certainty | 1.000 | 1.000 | 0.000 |
| credibility | 1.000 | 1.000 | 0.000 |
| moral_valence | 1.000 | 1.000 | 0.000 |
| negation | 1.000 | 1.000 | 0.000 |

`exfiltration` −5pp at peak; recovered to within 1pp by CAZ ensemble. Selective entanglement,
not broad collapse.

### Phase C — Direction decomposition

| Quantity | Value |
|----------|-------|
| R² of `d_ilya` explained by 17 labeled concepts | **6.07%** |
| Residual in `d_unknown` | **93.93%** |
| cos(`d_ilya`, `d_unknown`) | 0.969 |
| cos(`d_ilya`, `d_behav` — behavioral refusal) | **−0.13** (near-orthogonal) |

`d_ilya` and `d_behav` are two distinct objects extracted by two different methods. Prior
evasion work using behaviorally-extracted refusal directions is not testing IlyaGusev's method.

### Phase C — Surgical removal of `d_unknown`

Retention = S_surgery / S_baseline (Fisher separation).

| Concept | Baseline S | peak_only ret% | gem_targeted ret% | IlyaGusev full ret% |
|---------|-----------|----------------|-------------------|---------------------|
| causation | 1942 | 99.81% | 99.07% | 98.96% |
| certainty | 2692 | 100.00% | 99.93% | 99.69% |
| credibility | 6195 | 100.00% | 100.00% | 99.98% |
| moral_valence | 1392 | 99.97% | 97.34% | 93.93% |
| negation | 2047 | 100.00% | 100.00% | 100.03% |

`peak_only` = surgery at L24 only; `gem_targeted` = surgery at all GEM assembly-event layers.
`d_unknown` surgery is **more concept-preserving than IlyaGusev's full abliteration** on all
tested concepts.

Refusal separation remains ~13,000–14,000 under both surgery conditions — the refusal
*representation* survives; the behavioral change is routing disruption, not representation erasure.

---

## Interpretation

Three conditions Neural Chameleons' result would require:

| Condition | Required | Observed |
|-----------|----------|----------|
| LoRA suppresses probe representations on intact model | Yes | No — stuck at 0.99 |
| Abliteration broadly degrades safety-concept geometry | Yes | No — 9/10 zero-delta |
| `d_ilya` is a semantic concept direction | Yes (implied) | No — 6% explained by 17 concepts |

**Their result appears to require their specific abliterated-model substrate.** The abliteration
direction is not semantic, but its removal from L24 disrupts routing in ways that may undermine
LoRA gradient signal rather than the representations themselves.

---

## Data

Results at `~/rosetta_data/results/concept_evasion/` (also on HF `james-ra-henry/Rosetta-Activations`):

```
B_caz/
  google_gemma-2-9b-it/          — baseline Gemma CAZ profiles
  IlyaGusev_gemma-2-9b-it-abliterated/
  Qwen_Qwen2.5-7B-Instruct/
  gemma_d_unknown_peak_only/     — d_unknown surgery at L24 only
  gemma_d_unknown_gem_targeted/  — d_unknown surgery at all GEM layers
B_probes/                        — trained probe pkl + metrics
A_probes/                        — fixed-layer Phase A probes
A_eval/, A_triggered/            — Phase A evasion results
C_weight_diff/
  google_gemma-2-9b-it_vs_IlyaGusev_gemma-2-9b-it-abliterated/
    d_unknown_L24.npy            — orthogonal residual direction
    direction_decomposition_L24.json
    d_unknown_alllayer_summary.json
    refusal_direction.npy        — d_ilya (weight-diff SVD)
```

---

## Running

```bash
# Phase A
python scripts/A1_generate_data.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/A2_train_probes.py  --model Qwen/Qwen2.5-7B-Instruct
python scripts/A3_train_evasion.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/A4_eval_evasion.py  --model Qwen/Qwen2.5-7B-Instruct

# Phase B
python scripts/B1_extract_caz.py   --model Qwen/Qwen2.5-7B-Instruct
python scripts/B2_train_caz_probes.py --model Qwen/Qwen2.5-7B-Instruct
# B3 requires Neural Chameleons' unreleased training data

# Phase C
python scripts/C1_weight_diff.py   --donor google/gemma-2-9b-it \
                                   --target IlyaGusev/gemma-2-9b-it-abliterated
python scripts/C5_extract_behavioral_direction.py --model google/gemma-2-9b-it
python scripts/C6_compute_d_unknown.py            # GPU host; requires paper_n250 activations
python scripts/C2_transfer_abliteration.py        # surgery; see --help for modes
```

## Dependencies

```
torch>=2.0
transformers>=4.40
peft>=0.9
scikit-learn
numpy
```

`rosetta_tools` from `~/rosetta_tools/` (GPU host) or `~/Source/Rosetta_Program/rosetta_tools/` (dev)

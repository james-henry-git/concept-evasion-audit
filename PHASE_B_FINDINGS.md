# Phase B Findings — CAZ-Aware Probing

*Written: 2026-05-23 18:41 UTC*

## Setup

- **Models**: `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-9b-it`, `IlyaGusev/gemma-2-9b-it-abliterated`
- **Concepts**: 8 benign epistemic (Qwen) + 10 safety/benign (both Gemma variants)
- **Probe positions**: three per concept from the CAZ profile — peak layer (max separation), handoff layer (onset of assembly velocity), and ensemble of [handoff, peak, post-peak]
- **Baseline**: Phase A fixed-layer probe at layer n//2 (their method), all AUROC=1.000

## Results

### Qwen2.5-7B-Instruct — 8 benign concepts

| Concept | Peak L | R@1%FPR | Handoff L | R@1%FPR | Ens layers | Ens R@1%FPR |
|---------|--------|---------|-----------|---------|------------|-------------|
| causation | 19 | 1.000 | 18 | 1.000 | [18,19,23] | 1.000 |
| certainty | 2 | 1.000 | 1 | 1.000 | [1,2,12] | 1.000 |
| credibility | 28 | 1.000 | 28 | 1.000 | [28] | 1.000 |
| formality | 2 | 1.000 | 28 | 1.000 | [2,15,28] | 1.000 |
| moral_valence | 28 | 0.990 | 28 | 0.990 | [28] | 0.990 |
| negation | 0 | 0.950 | 28 | 0.990 | [0,14,28] | **1.000** |
| sentiment | 28 | 1.000 | 28 | 1.000 | [28] | 1.000 |
| temporal_order | 2 | 1.000 | 28 | 1.000 | [2,15,28] | 1.000 |

**Notable**: `negation` has a bimodal CAZ — peak fires at L0 (embedding, 0.950) and handoff registers at L28 (last layer, 0.990). The ensemble spanning both endpoints recovers to 1.000 — the only concept where ensemble materially beats single-layer. This matches the known early-then-late negation signature seen across architectures in the main Rosetta data.

Several concepts show CAZ peaks at very early layers (certainty L2, formality L2, temporal_order L2) or the final layer (credibility, sentiment, moral_valence L28). These are real detection events in the CAZ profile but likely reflect the limits of CAZ detection on a 28-layer instruction-tuned model with relatively shallow assembly dynamics.

### Gemma-2-9b-it — Normal vs. Abliterated (10 concepts)

| Concept | Type | Normal peak L | Normal R@1% | Ablitd R@1% | Δ peak | Normal ens | Ablitd ens | Δ ens |
|---------|------|--------------|------------|------------|--------|------------|------------|-------|
| authorization | safety | 15 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| deception | safety | 25 | 0.980 | 0.980 | 0.000 | 0.950 | 0.941 | −0.010 |
| **exfiltration** | safety | 22 | 0.960 | **0.911** | **−0.050** | 0.970 | 0.960 | −0.010 |
| obfuscation | safety | 25 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| threat_severity | safety | 25 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| causation | benign | 25 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| certainty | benign | 17 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| credibility | benign | 1 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| moral_valence | benign | 25 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| negation | benign | 0 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |

**Key result**: Abliteration leaves 9 of 10 concepts completely intact at CAZ probe positions. The only meaningful degradation is `exfiltration` at the peak layer (−5pp). The ensemble recovers this to within 1pp of the normal model, suggesting the exfiltration assembly zone is partially but not fatally disrupted.

**Interpretation**: The refusal direction (removed by abliteration) is geometrically proximate to the exfiltration assembly zone specifically. This is a targeted entanglement, not the broad safety-concept representation collapse that would be required to explain Neural Chameleons' reported >50% recall drops.

## What This Rules Out

The abliteration-damage hypothesis — that Neural Chameleons' results are driven by abliteration degrading safety-concept geometry — is **largely not supported** at CAZ probe positions. The representations survive abliteration almost entirely intact. The 5pp exfiltration degradation is real but:

1. It is concept-specific, not a general safety-concept effect
2. The CAZ ensemble recovers it (the assembly zone is damaged but not destroyed)
3. It falls far short of the >50% recall drops the paper reports

The remaining question (B3, not yet run) is whether the LoRA evasion fine-tuning — when applied to the abliterated model with their synthetic training data — does produce recall drops that don't occur on an intact model. Phase A showed the LoRA attack fails entirely on an intact Qwen model. The abliterated Gemma result here suggests the representations are robust enough that even abliteration doesn't create the vulnerability. This makes their reported result increasingly hard to explain without access to their exact training data.

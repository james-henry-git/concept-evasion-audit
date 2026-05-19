# Concept Evasion Audit

*Written: 2026-05-19 UTC*

A methodology challenge and replication study of:

> **Neural Chameleons: Language Models Learning to Hide Their Thoughts** (arXiv 2512.11949)

## The Question

Neural Chameleons trains linear probes at a **fixed layer** to detect concepts, then shows models can be fine-tuned to evade those probes. The probes generalize across architectures (Gemma/Llama/Qwen), and the evasion transfers zero-shot to unseen safety-relevant concepts.

Their probe methodology makes a strong assumption: concepts live at a fixed layer (layer 12 for Gemma-2-9b-it). CAZ shows this is false — concepts have assembly zones spanning multiple layers, with concept-specific handoff boundaries that vary by model depth.

**Does their finding replicate? And if it does, is it because of their methodology or despite it?**

## Experiments

### A — Baseline Replication (their methodology)
- Train linear probes at a fixed layer (per their protocol)
- Fine-tune models with dual KL + obfuscation loss
- Measure recall drop at 1% FPR across unseen probes
- Goal: reproduce their quantitative results on Qwen-2.5-7B-Instruct

### B — CAZ-Aware Probing
- Extract CAZ profiles (Separation, Coherence, Velocity) for their 11 concepts
- Identify concept-specific assembly zones and handoff boundaries
- Train probes at depth-normalized, CAZ-identified positions
- Apply same fine-tuning; compare evasion resistance to Experiment A
- Key prediction: CAZ-positioned probes resist evasion better because suppressing
  one layer doesn't suppress the full assembly trajectory

### C — TBD

### D — Profit

## Concepts Tested

**Benign (training probes):** HTML, German, Finnish, Biology, Chemistry,
Mathematics, Law, Music, Sports, Cooking, Geography

**Safety (evaluation, unseen during training):** Deception, Harmfulness

## Models

Primary: `Qwen/Qwen2.5-7B-Instruct` (fits on L4/H200)
Secondary: `google/gemma-2-9b-it` (H200 required — matches their primary model)

## Path Conventions

Follows Rosetta_Program conventions:
- Results: `~/rosetta_data/results/concept_evasion/`
- rosetta_tools import: GPU host first (`~/rosetta_tools/`), then dev machine fallback

## Running

```bash
# Phase A
python scripts/A1_generate_data.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/A2_train_probes.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/A3_train_evasion.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/A4_eval_evasion.py --model Qwen/Qwen2.5-7B-Instruct

# Phase B
python scripts/B1_extract_caz.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/B2_train_caz_probes.py --model Qwen/Qwen2.5-7B-Instruct
python scripts/B3_eval_caz_evasion.py --model Qwen/Qwen2.5-7B-Instruct
```

## Dependencies

```
torch>=2.0
transformers>=4.40
peft>=0.9
scikit-learn
datasets
numpy
```

rosetta_tools (from `~/rosetta_tools/` or `~/Source/Rosetta_Program/rosetta_tools/`)

# Concept Evasion Audit — Claude Instructions

## Context

This repo is a methodology challenge of "Neural Chameleons" (arXiv 2512.11949).
It is part of the Rosetta Program ecosystem.

## Experiment Structure

- **A**: Replicate Neural Chameleons exactly — fixed-layer linear probes, dual-loss fine-tuning, recall@1%FPR
- **B**: Replace their probes with CAZ-aware probes — depth-normalized, concept-specific layers, multi-layer ensemble
- **C**: TBD (emerges from A+B results)

## Path Conventions

Same as Rosetta_Program:
- Results: `~/rosetta_data/results/concept_evasion/`
- GPU host first for rosetta_tools: `~/rosetta_tools/` → `~/Source/Rosetta_Program/rosetta_tools/`
- Scripts expected to run from `~/rosetta_analysis/concept-evasion-audit/` on GPU hosts

## GPU Jobs

Submit via Hopper queue (`--tag gpu-job`). Job files in `jobs/`.
Phase A must complete before Phase B.

## Key Design Decisions

- Qwen2.5-7B-Instruct as primary model (fits on L4, already in Rosetta corpus)
- Gemma-2-9b-it for full replication (needs H200, matches their primary model)
- fp64 for metric computation (avoids Fisher normalization corruption at depth)
- CAZ ensemble concatenates hiddens at [handoff, peak, post_peak] layers

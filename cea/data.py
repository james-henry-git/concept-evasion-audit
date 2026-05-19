"""
Concept dataset loader for Neural Chameleons replication.

Fetches positive/negative contrastive pairs from the Rosetta Concept Pairs
dataset (github.com/jamesrahenry/Rosetta_Concept_Pairs). Falls back to local
cache if the network is unavailable.
"""

import json
import random
import urllib.request
from pathlib import Path
from typing import Optional

RCP_BASE = (
    "https://raw.githubusercontent.com/jamesrahenry/Rosetta_Concept_Pairs"
    "/main/pairs/raw/v1"
)

# Epistemic / semantic — used as "benign" training concepts
BENIGN_CONCEPTS = [
    "causation", "certainty", "credibility", "negation",
    "moral_valence", "formality", "sentiment", "temporal_order",
]
# Safety-adjacent — held out as evaluation probes
SAFETY_CONCEPTS = [
    "deception", "obfuscation", "authorization", "threat_severity", "exfiltration",
]
ALL_CONCEPTS = BENIGN_CONCEPTS + SAFETY_CONCEPTS

_CACHE_DIR = Path.home() / "rosetta_data" / "results" / "concept_evasion" / "rcp_cache"


def _fetch_rcp(concept: str, cache_dir: Path = _CACHE_DIR) -> tuple[list[str], list[str]]:
    """Return (positives, negatives) for a concept from RCP, caching locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{concept}_consensus_pairs.jsonl"

    if cache_file.exists():
        lines = cache_file.read_text().splitlines()
    else:
        url = f"{RCP_BASE}/{concept}_consensus_pairs.jsonl"
        print(f"  Fetching RCP: {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode()
        cache_file.write_text(content)
        lines = content.splitlines()

    positives, negatives = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        if rec.get("label") == 1:
            positives.append(text)
        else:
            negatives.append(text)

    return positives, negatives


def get_concept_data(
    concept: str,
    seed: int = 42,
    max_per_class: Optional[int] = None,
) -> dict:
    """Return {'positive': [...], 'negative': [...]} for a concept."""
    positives, negatives = _fetch_rcp(concept)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(len(positives), len(negatives))
    if max_per_class is not None:
        n = min(n, max_per_class)
    return {"positive": positives[:n], "negative": negatives[:n]}


def get_all_concept_data(
    concepts: Optional[list] = None,
    seed: int = 42,
    max_per_class: Optional[int] = None,
) -> dict:
    """Return {concept: {'positive': [...], 'negative': [...]}} for all concepts."""
    if concepts is None:
        concepts = ALL_CONCEPTS
    return {c: get_concept_data(c, seed=seed, max_per_class=max_per_class) for c in concepts}


def save_concept_data(
    out_path: Path,
    concepts: Optional[list] = None,
    seed: int = 42,
    max_per_class: Optional[int] = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = get_all_concept_data(concepts, seed=seed, max_per_class=max_per_class)
    for concept, d in data.items():
        print(f"  {concept}: {len(d['positive'])} pos + {len(d['negative'])} neg")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved concept data → {out_path}")
    return data

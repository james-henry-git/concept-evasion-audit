#!/usr/bin/env python3
"""
A1: Fetch concept datasets from Rosetta Concept Pairs and save to disk.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cea import RESULTS_DIR
from cea.data import save_concept_data, BENIGN_CONCEPTS, SAFETY_CONCEPTS, ALL_CONCEPTS

parser = argparse.ArgumentParser()
parser.add_argument("--concepts", nargs="+", default=None)
parser.add_argument("--max-per-class", type=int, default=None,
                    help="Cap positive and negative examples per concept (default: all)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

concepts = args.concepts or ALL_CONCEPTS
out_path = RESULTS_DIR / "concept_data.json"
save_concept_data(out_path, concepts=concepts, seed=args.seed, max_per_class=args.max_per_class)
print(f"Concepts: {concepts}")
print(f"Benign: {[c for c in concepts if c in BENIGN_CONCEPTS]}")
print(f"Safety: {[c for c in concepts if c in SAFETY_CONCEPTS]}")

from pathlib import Path

RESULTS_DIR = Path.home() / "rosetta_data" / "results" / "concept_evasion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# rosetta_tools import with GPU-host-first fallback
import sys
_rt_gpu = Path.home() / "rosetta_tools"
_rt_dev = Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"
_rt_path = _rt_gpu if _rt_gpu.exists() else _rt_dev
if str(_rt_path.parent) not in sys.path:
    sys.path.insert(0, str(_rt_path.parent))

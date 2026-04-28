"""
Add each script directory to sys.path so tests can import them directly.
"""
import sys
from pathlib import Path

_ML_DIR = Path(__file__).parent.parent

for _subdir in ("data_ingest", "anomaly_detection", "evaluation"):
    _p = str(_ML_DIR / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

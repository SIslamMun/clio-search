#!/usr/bin/env python3
"""L2-B Large Scale: CLIO + IOWarp at 50K and 100K blobs.

Same test as L2_clio_iowarp_integration.py but only runs
the larger scales with extended timeout (7200s per run).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Import the main script
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "eval" / "eval_final" / "code" / "laptop"))

# Monkey-patch the scales and timeout before importing
import L2_clio_iowarp_integration as L2

L2.SCALES = [50_000, 100_000]

# Also need to patch the timeout in the main function
# Read the source and find the timeout
import inspect
source = inspect.getsource(L2.main)

if __name__ == "__main__":
    # Override timeout by modifying the subprocess call
    import subprocess
    _orig_run = subprocess.run

    def _patched_run(*args, **kwargs):
        if "timeout" in kwargs and kwargs["timeout"] == 3600:
            kwargs["timeout"] = 14400  # 4 hours
            print(f"[PATCH] Extended Docker timeout to {kwargs['timeout']}s")
        return _orig_run(*args, **kwargs)

    subprocess.run = _patched_run
    L2.main()

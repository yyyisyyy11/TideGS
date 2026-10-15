#!/usr/bin/env python3
"""Historical PC1 entry point; active implementation uses Micro Pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_micro_pipeline_overlap import *  # noqa: F401,F403

if __name__ == "__main__":
    main()

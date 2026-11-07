#!/usr/bin/env python3
"""Historical PC1 entry point; active implementation uses Micro Pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_micro_pipeline_submission_order import *  # noqa: F401,F403

if __name__ == "__main__":
    # Preserve the historical default even when new and old CSVs coexist.
    if not any(arg == "--prefix" or arg.startswith("--prefix=") for arg in sys.argv[1:]):
        sys.argv.extend(["--prefix", "pc1-overlap"])
    main()

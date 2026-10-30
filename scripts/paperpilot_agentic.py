"""Compatibility launcher for subprocess and recovery tests.

The installed ``paperpilot`` console script is the preferred user entry point.
Keeping this tiny launcher makes a checkout directly executable before install and
preserves cross-process checkpoint tests from the original runtime.
"""

from pathlib import Path
import sys


# The script name matches the package for backwards compatibility, so put the
# source tree ahead of this directory to avoid importing this file as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_agentic.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

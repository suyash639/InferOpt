"""Package executable entrypoint for inferopt.benchmarks."""

import sys

from inferopt.benchmarks.cli import main

if __name__ == "__main__":
    sys.exit(main())

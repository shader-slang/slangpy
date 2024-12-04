# SPDX-License-Identifier: Apache-2.0

from fix_license import process_file
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python local_precommit.py <files>")
        sys.exit(1)
    for fn in sys.argv[1:]:
        process_file(Path(fn))

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Move, resize, or maximize the window, close it, then run again to restore it.

Run with: python examples/window/window_placement.py
Delete window_placement.json beside this script to reset the saved placement.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Optional

import slangpy as spy


def load_placement(path: Path) -> Optional[spy.WindowPlacement]:
    if not path.exists():
        return None
    try:
        return spy.WindowPlacement(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Could not load {path}: {error}. Using the default placement.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()

    window = spy.Window(
        width=800,
        height=600,
        title="SlangPy placement example - close and reopen to restore",
        placement=load_placement(args.state_file),
    )
    print("Move, resize, or maximize the window, then close it and run this script again.")
    print("Ctrl+C also saves and exits, including while the window is minimized.")

    try:
        while not window.should_close():
            window.process_events()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # Read placement before hiding or destroying the window. It retains the
        # normal geometry and maximization state even when currently minimized.
        args.state_file.write_text(
            json.dumps(window.placement.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved placement to {args.state_file}")
        window.hide()
        window.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate an episode-index -> success-label JSON mapping.

The generated file is intentionally a JSON object (rather than a list) so it
can be consumed directly by ``add_returns_to_lerobot.py``'s ``--success-labels``
option.  The range is inclusive: ``--start 0 --end 14346`` creates 14,347
entries with keys ``"0"`` through ``"14346"``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_OUTPUT = "/inspire/hdd/global_user/feisenyu-253108140203/datasets/libero_plus_lerobot/success_labels.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="First episode index (inclusive).")
    parser.add_argument("--end", type=int, default=14346, help="Last episode index (inclusive).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--failure",
        action="store_true",
        help="Write false labels instead of true labels (useful for a negative-control file).",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact output.")
    return parser.parse_args()


def write_labels(start: int, end: int, output: Path, value: bool = True, indent: int = 2) -> int:
    """Atomically write labels for the inclusive episode range and return its size."""
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")
    if indent < 0:
        raise ValueError(f"indent must be non-negative, got {indent}")

    labels = {str(index): bool(value) for index in range(start, end + 1)}
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write beside the destination and replace it atomically, preventing a
    # partially-written mapping if the process is interrupted.
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(labels, handle, ensure_ascii=False, indent=(indent or None), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return len(labels)


def main() -> None:
    args = _parse_args()
    count = write_labels(args.start, args.end, args.output, value=not args.failure, indent=args.indent)
    label = "false" if args.failure else "true"
    print(f"Wrote {count} labels ({label}) for episodes {args.start}..{args.end} to {args.output}")


if __name__ == "__main__":
    main()

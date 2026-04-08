"""Log shuffler — extracts blocks from a log file, shuffles them, and reassembles
into a new stream of configurable length. Each call produces a different ordering,
giving the demo infinite variety from a fixed dataset."""

from __future__ import annotations

import random
import re
from pathlib import Path

_BLOCK_RE = re.compile(r"(blk_-?\d+)")


def _extract_blocks(filepath: Path) -> dict[str, list[str]]:
    """Read a log file and group lines by block ID.

    Lines with no block ID are attached to a synthetic ``_unassigned`` key
    so they are not silently dropped.

    Returns:
        Ordered dict mapping block_id → list of raw log lines.
    """
    blocks: dict[str, list[str]] = {}
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip()
            if not line:
                continue
            m = _BLOCK_RE.search(line)
            key = m.group(1) if m else "_unassigned"
            blocks.setdefault(key, []).append(line)
    return blocks


def shuffled_stream(
    filepath: str | Path,
    n_lines: int = 100,
    seed: int | None = None,
) -> list[str]:
    """Return a shuffled log stream of up to ``n_lines`` lines.

    Blocks are shuffled as atomic units (internal event order is preserved
    within each block). Blocks are sampled without replacement until
    ``n_lines`` is reached or all blocks are exhausted.

    Args:
        filepath: Path to the source log file.
        n_lines: Target number of output lines (soft limit — the last block
            included may push the total slightly above this value).
        seed: Optional random seed for reproducible shuffles (useful in tests).

    Returns:
        List of raw log line strings in shuffled block order.
    """
    filepath = Path(filepath)
    blocks = _extract_blocks(filepath)

    block_ids = [k for k in blocks if k != "_unassigned"]
    rng = random.Random(seed)
    rng.shuffle(block_ids)

    result: list[str] = []
    for block_id in block_ids:
        if len(result) >= n_lines:
            break
        result.extend(blocks[block_id])

    return result

"""Fetch one MOTChallenge sequence, so accuracy can be scored against real labels.

The official host, motchallenge.net, resolves but refuses connections — an
independent proxy times out against it too, so it is down rather than blocked.
The same archives are mirrored on the Hugging Face hub in their original layout,
and a mirror is fine here because the file that matters is a plain text list of
hand-drawn boxes that can be inspected on arrival.

One sequence is enough to start: MOT17 ships each scene three times over, once
per public detector, and the images are identical between them. We run our own
detector, so the ``-FRCNN`` copy is taken purely as a naming convention.

Usage: python scripts/fetch_mot.py [--sequence MOT17-02-FRCNN] [--frames N]
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

_REPO = "Morrison1025/MOT17"
_BASE = f"https://huggingface.co/datasets/{_REPO}/resolve/main/train"
_DEST = Path("data/mot")


def _get(url: str, into: Path, quiet: bool = False) -> bool:
    """Download one file, skipping it when already present and non-empty."""
    if into.exists() and into.stat().st_size > 0:
        return True
    into.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        if not quiet:
            print(f"  failed {url}: {exc}")
        return False
    into.write_bytes(data)
    return True


def _sequence_length(seqinfo: Path) -> int:
    for line in seqinfo.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("seqlength"):
            return int(line.split("=", 1)[1])
    raise ValueError(f"{seqinfo} does not state seqLength")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="MOT17-02-FRCNN")
    parser.add_argument("--frames", type=int, default=None,
                        help="stop after N frames; the labels still cover the whole sequence")
    parser.add_argument("--dest", type=Path, default=_DEST)
    args = parser.parse_args()

    target = args.dest / args.sequence
    print(f"{args.sequence} -> {target}")

    # Labels and metadata first: without these the images are of no use, and
    # failing here means the mirror is wrong before anything large is pulled.
    for name in ("seqinfo.ini", "gt/gt.txt"):
        if not _get(f"{_BASE}/{args.sequence}/{name}", target / name):
            raise SystemExit(f"could not fetch {name}; the mirror may have moved")

    length = _sequence_length(target / "seqinfo.ini")
    wanted = length if args.frames is None else min(args.frames, length)
    print(f"  {length} frames labelled, fetching {wanted}")

    got = 0
    for number in range(1, wanted + 1):
        name = f"img1/{number:06d}.jpg"
        if _get(f"{_BASE}/{args.sequence}/{name}", target / name):
            got += 1
        else:
            break
        if got % 50 == 0:
            print(f"  {got}/{wanted}", flush=True)

    if got < wanted:
        raise SystemExit(f"stopped after {got} frames")

    # A sequence whose seqinfo promises more frames than were fetched would make
    # the reader raise part-way through a run, so it is corrected to what exists.
    if got < length:
        text = (target / "seqinfo.ini").read_text(encoding="utf-8")
        text = "\n".join(
            f"seqLength={got}" if line.lower().startswith("seqlength") else line
            for line in text.splitlines()
        )
        (target / "seqinfo.ini").write_text(text + "\n", encoding="utf-8")

    print(f"  done: {got} frames")
    print(f"\nScore it with:\n  python scripts/mot_eval.py --data {args.dest}")


if __name__ == "__main__":
    main()

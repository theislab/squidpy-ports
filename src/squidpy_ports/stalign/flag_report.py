"""Read a two-condition flag sweep and say which metrics moved by more than the noise.

Written for ledger row D13 and usable for any comparison-only flag: point it at the root a
flag job wrote and it reports, per notebook and per metric, the effect against the
within-condition spread.

The verdict column exists because D11's first measurement did not have one. With one
replicate per condition the rank-3 region disagreement looked like 49.8 % -> 18.1 %; the
second replicate came back at 42.2 %, so the effect (0.194) was smaller than the spread
(0.241) and the whole reading evaporated. An effect is only readable here when it exceeds
the spread of the two conditions it sits between, and this prints both rather than leaving
that comparison to whoever reads the table.

Expects the layout :file:`{condition}-rep{n}/{notebook}-metrics.json`, which is what
``.claude/run_stalign_d13_flag.sbatch`` writes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

#: `<condition>-rep<n>`, the directory name one launcher invocation was given.
_RUN_DIRECTORY = re.compile(r"^(?P<condition>[a-z0-9-]+)-rep(?P<rep>\d+)$")


def read_runs(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """``{notebook: {condition: {metric: [values...]}}}`` gathered off disk.

    Replicates of one condition are collected into a list per metric, in replicate order, so
    the spread is computed from the same numbers the table prints.
    """
    runs: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for directory in sorted(root.iterdir()):
        match = _RUN_DIRECTORY.match(directory.name) if directory.is_dir() else None
        if match is None:
            continue
        for path in sorted(directory.glob("*-metrics.json")):
            notebook = path.name.removesuffix("-metrics.json")
            for metric, value in json.loads(path.read_text()).items():
                if isinstance(value, (int, float)):
                    runs[notebook][match["condition"]][metric].append(float(value))
    return runs


def effect_and_spread(baseline: list[float], treatment: list[float]) -> tuple[float, float]:
    """The change between conditions, and the largest disagreement within either of them.

    Median rather than mean across replicates: with two of them they coincide, and with more
    the median does not let one wandering replicate carry the estimate.
    """
    effect = abs(statistics.median(treatment) - statistics.median(baseline))
    spread = max(max(values) - min(values) for values in (baseline, treatment))
    return effect, spread


def verdict(effect: float, spread: float) -> str:
    """Whether the effect can be read at all, in the row's own words."""
    if spread == 0.0:
        return "measured (replicates identical)" if effect else "no change"
    if effect > spread:
        return f"measured ({effect / spread:.1f}x spread)"
    return "BELOW SPREAD -- not readable"


def _format(values: list[float]) -> str:
    return " / ".join(f"{value:.6g}" for value in values)


def report(runs: dict[str, dict[str, dict[str, list[float]]]], baseline: str, treatment: str) -> str:
    """One markdown section per notebook, worst-first by effect-over-spread."""
    lines: list[str] = []
    for notebook in sorted(runs):
        conditions = runs[notebook]
        missing = [name for name in (baseline, treatment) if name not in conditions]
        if missing:
            lines += [f"### `{notebook}`", "", f"Incomplete: no `{'`, `'.join(missing)}` run.", ""]
            continue
        rows = []
        for metric in sorted(conditions[baseline]):
            if metric not in conditions[treatment]:
                continue
            before, after = conditions[baseline][metric], conditions[treatment][metric]
            effect, spread = effect_and_spread(before, after)
            # Sort key, not a printed number: an effect with zero spread outranks every
            # effect that has to be discounted by one.
            rank = effect / spread if spread else float("inf") if effect else 0.0
            rows.append((rank, metric, _format(before), _format(after), effect, spread))
        lines += [
            f"### `{notebook}`",
            "",
            f"| metric | {baseline} (reps) | {treatment} (reps) | effect | spread | verdict |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _rank, metric, before, after, effect, spread in sorted(rows, reverse=True):
            lines.append(
                f"| `{metric}` | {before} | {after} | {effect:.6g} | {spread:.6g} | {verdict(effect, spread)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Render the flag sweep's table to stdout, or to ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding the `<condition>-rep<n>` runs")
    parser.add_argument("--baseline", default="baseline", help="condition run with the flag off")
    parser.add_argument("--treatment", default="collapsed", help="condition run with the flag on")
    parser.add_argument("--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args()

    runs = read_runs(args.root)
    if not runs:
        raise SystemExit(f"No `<condition>-rep<n>/*-metrics.json` found under {args.root}")
    text = report(runs, args.baseline, args.treatment)
    if args.output:
        args.output.write_text(text + "\n")
        print(f"### wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

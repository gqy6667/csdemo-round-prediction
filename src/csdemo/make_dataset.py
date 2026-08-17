from __future__ import annotations

import argparse
from pathlib import Path

from .features import make_first_kill_samples, make_pre_round_samples
from .io import read_table, write_table
from .quality import evaluate_quality, raise_for_errors, write_quality_report
from .split import add_group_split


def find_table(input_dir: str | Path, stem: str) -> Path:
    input_dir = Path(input_dir)
    candidates = []
    for suffix in (".parquet", ".csv", ".jsonl", ".json"):
        candidates.extend(input_dir.glob(f"**/{stem}{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {stem}.parquet/csv/json/jsonl under {input_dir}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing normalized rounds and kills tables.")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--quality-report-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    rounds = read_table(find_table(input_dir, "rounds"))
    kills = read_table(find_table(input_dir, "kills"))
    summary, examples = evaluate_quality(rounds, kills)
    if args.quality_report_dir:
        write_quality_report(summary, examples, args.quality_report_dir)
    raise_for_errors(summary)

    pre_round = add_group_split(make_pre_round_samples(rounds))
    first_kill = add_group_split(make_first_kill_samples(rounds, kills))

    write_table(pre_round, output_dir / f"pre_round.{args.format}")
    write_table(first_kill, output_dir / f"first_kill.{args.format}")

    print(f"Wrote {len(pre_round):,} pre-round samples")
    print(f"Wrote {len(first_kill):,} first-kill samples")


if __name__ == "__main__":
    main()

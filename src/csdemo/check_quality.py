from __future__ import annotations

import argparse

from .io import read_table
from .make_dataset import find_table
from .quality import evaluate_quality, raise_for_errors, write_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized ESTA rounds and kills tables.")
    parser.add_argument("--input", required=True, help="Folder containing rounds and kills tables.")
    parser.add_argument("--report-dir", required=True, help="Folder for quality_summary.csv and quality_examples.csv.")
    args = parser.parse_args()

    rounds = read_table(find_table(args.input, "rounds"))
    kills = read_table(find_table(args.input, "kills"))
    summary, examples = evaluate_quality(rounds, kills)
    write_quality_report(summary, examples, args.report_dir)

    if summary.empty:
        print("Quality check passed with no findings.")
        return

    print(summary.to_string(index=False))
    raise_for_errors(summary)


if __name__ == "__main__":
    main()

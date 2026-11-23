"""CLI entry point: python -m russian_rates."""

import argparse
from datetime import date
from pathlib import Path

from .cbr import SOURCES, fetch_series, parse_date, save_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Bank of Russia rates to Parquet")
    parser.add_argument("--series", choices=["all", *SOURCES], default="all")
    parser.add_argument("--start", type=parse_date, default=date(2014, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date.today())
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--replace", action="store_true", help="Overwrite existing history instead of merging")
    parser.add_argument("--pause", type=float, default=0.2, help="Pause between yearly requests")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must not be after --end")
    if args.pause < 0:
        parser.error("--pause must be non-negative")
    selected = list(SOURCES) if args.series == "all" else [args.series]
    for series in selected:
        print(f"Downloading {series}: {args.start} through {args.end}...", flush=True)
        frame = fetch_series(series, args.start, args.end, pause_seconds=args.pause)
        path = save_parquet(frame, series, args.output_dir, replace=args.replace)
        print(f"  {len(frame)} records downloaded; saved to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

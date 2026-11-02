"""Command-line entry point for the reproducible paper analyses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from . import __version__
from .data import Dataset
from .validation import validate_data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Audit and calibrate self-reported peer-review data.")
    parser.add_argument("command", choices=("reproduce", "analyze", "figures", "validate"),
                        nargs="?", default="reproduce")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Repository root, data folder, or processed CSV folder (default: bundled data)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Output folder (default: outputs in the current working directory)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap replicates per comparison; use 0 to skip (default: 1000)")
    parser.add_argument("--seed", type=int, default=20261101, help="Base bootstrap seed")
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output = args.output_dir.resolve()
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    validation = validate_data(args.data_dir)
    (tables / "data_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(f"Data checked: {validation['total_reports_represented']:,} reports across {validation['events']} events.", flush=True)
    if args.command == "validate":
        print(f"Saved {tables / 'data_validation.json'}")
        return

    from .analysis import run_analysis
    iterations = args.bootstrap if args.command != "figures" else 0
    print("Recomputing descriptive and calibration results...", flush=True)
    if iterations:
        print(f"Bootstrap: {iterations:,} replicates for each of six comparisons.", flush=True)
    results = run_analysis(args.data_dir, tables, bootstrap_iterations=iterations, seed=args.seed)
    if args.command in {"reproduce", "figures"}:
        from .figures import generate_figures
        print("Drawing four main and four supplementary figures...", flush=True)
        paths = generate_figures(args.data_dir, output / "figures", results=results)
        print(f"Saved {len(paths)} figures as PDF and PNG.", flush=True)

    folder = Dataset(args.data_dir).data_dir
    info = {
        "package_version": __version__, "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
        "command": args.command, "bootstrap_iterations": iterations, "seed": args.seed,
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.csv"))},
    }
    (output / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Results: {output}", flush=True)


if __name__ == "__main__":
    main()

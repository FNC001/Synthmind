"""Command-line entry points for Synthmind V1.0."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from synthmind import __version__
from synthmind.release_layout import check_data_root


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_workflow(args: argparse.Namespace) -> int:
    command = [
        "bash",
        str(repository_root() / "scripts" / "run_v1_workflow.sh"),
        "--data-root",
        str(args.data_root),
        "--work-root",
        str(args.work_root),
        "--python",
        str(args.python),
        "--profile",
        args.profile,
        "--device",
        args.device,
    ]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, check=False).returncode


def check_data(args: argparse.Namespace) -> int:
    report = check_data_root(args.data_root, profile=args.profile)
    for item in report["checks"]:
        marker = "OK" if item["exists"] else "MISSING"
        print(f"{marker:7} {item['path']}")
    print(
        f"{report['present']}/{report['required']} required paths present "
        f"for profile={args.profile}"
    )
    return 0 if report["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthmind")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workflow = subparsers.add_parser("workflow", help="run the V1.0 workflow")
    workflow.add_argument("--data-root", type=Path, required=True)
    workflow.add_argument("--work-root", type=Path, default=Path("work/v1"))
    workflow.add_argument("--python", default=sys.executable)
    workflow.add_argument(
        "--profile",
        choices=("validate-full", "validate-fast", "train", "rebuild-train"),
        default="validate-full",
    )
    workflow.add_argument("--device", default="cuda")
    workflow.add_argument("--dry-run", action="store_true")
    workflow.set_defaults(handler=run_workflow)

    data = subparsers.add_parser("check-data", help="check an external data root")
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument(
        "--profile",
        choices=("validate-full", "validate-fast", "train", "rebuild-train"),
        default="validate-full",
    )
    data.set_defaults(handler=check_data)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

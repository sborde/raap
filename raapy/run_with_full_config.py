"""
This is the entry point of the analyses started from a full configuration file.

This functionality might be converted to multiple Jupyter notebooks.
"""

from argparse import ArgumentParser

from workflows.common.file_io import load_yaml
from workflows.spike_description import run_slowdown_analysis

COMMANDS = {
    'slowdown_analysis': run_slowdown_analysis
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run an analysis defined by a config file."
    )
    parser.add_argument(
        'command',
        type=str,
        help="Name of the analysis to run."
    )
    args = parser.parse_args()

    config = load_yaml(f"{args.command}.yaml")

    if args.command not in COMMANDS:
        raise ValueError(f"Unknown command: {args.command}")

    COMMANDS[args.command](config)

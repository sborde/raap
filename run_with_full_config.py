
from argparse import ArgumentParser
import yaml

from run_full_slowdown_analysis import run_slowdown_analysis

ROUTING = {
    'slowdown_analysis': run_slowdown_analysis
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Run full slowdown analysis on dataset.'
    )
    parser.add_argument(
        'command',
        type=str,
        help='Path to the analysis configuration file.'
    )
    args = parser.parse_args()

    with open(f"{args.command}.yaml", 'r') as file:
        # Load the configuration file
        full_config = yaml.safe_load(file)

    ROUTING[args.command](full_config)



import abc
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Protocol

import yaml

from common.wrappers import CellPath
from .path_collectors import collect_paths

def load_config(path: str) -> dict:
    """
    Load the configuration from the specified path.
    Args:
        path (str): Path to the configuration file.
    Returns:
        dict: Loaded configuration dictionary.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def setup_result_directory(output_str: str) -> dict:
    """
    Prepare the result directory by creating necessary subdirectories.
    """
    output = Path(output_str)
    paths = {}
    for subdir in ['01_processed', '02_figures', '03_statistics']:
        paths[subdir] = (output / subdir)
        paths[subdir].mkdir(parents=True, exist_ok=True)
    return paths

def create_default_argument_parser() -> ArgumentParser:
    """
    Create a default argument parser for all analysis scripts.
    It expects a configuration file path as an argument.
    """
    parser = ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='Path to the configuration file'
    )
    return parser

def collect_inputs(config: dict) -> list[CellPath]:
    """
    Collects input files based on the configuration.
    Args:
        config (dict): dictionary containing input configuration.
    Returns:
        list[CellPath]: List of input file paths.
    """
    return [
        CellPath(file_path, file_path.stem, file_path.parent.name)
        for file_path in collect_paths(config)
    ]

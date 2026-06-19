"""Module for analysis config file handling."""

from pathlib import Path

import yaml


def load_config(name: str) -> dict:
    """
    Loads the customized configuration for the analysis specified by its name.

    If no configuration has been given, raises an error. If only one config
    file exists, it is used. If both exist, they are merged, custom will override
    default.
    Args:
        name (str): Name of the configuration file (without extension).
    Returns:
        dict: Loaded configuration dictionary.
    """

    default = load_yaml(f'config_defaults/{name}.yaml')
    custom = load_yaml(f'config/{name}.yaml')

    # No configuration found
    if not default and not custom:
        raise FileNotFoundError(f"Configuration files for {name} not found.")

    # If only one configuration exists, return it
    if not custom:
        return default
    if not default:
        return custom

    # Merge configurations
    return merge_dicts(default, custom)

def load_yaml(path: str) -> dict:
    """
    Load a YAML file if it exists, otherwise return an empty dictionary.
    
    Args:
        path (str): Path to the YAML file.
        
    Returns:
        dict: Loaded configuration or an empty dictionary if the file does not exist.
    """
    if not Path(path).exists():
        return {}
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def merge_dicts(default: dict, custom: dict) -> dict:
    """
    Merges two dictionaries recursively, with values from the custom dictionary
    overriding those in the default dictionary.

    Args:
        default (dict): The default configuration dictionary.
        custom (dict): The custom configuration dictionary.

    Returns:
        dict: The merged configuration dictionary.
    """
    merged = default.copy()
    for key, value in custom.items():
        if key in merged and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

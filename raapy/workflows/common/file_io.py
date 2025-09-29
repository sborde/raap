"""Utility functions for input/output operations."""

import yaml
import json

def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def save_json(path: str, data: dict | list[dict]) -> None:
    """
    Save a dictionary as a JSON file at the specified path.
    
    Args:
        path (str): Path to save the JSON file.
        data (dict): Data to be saved in JSON format.
    """
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def load_textfile(path: str) -> str:
    """
    Load a text file from the specified path.
    
    Args:
        path (str): Path to the text file.

    Returns:
        str: Content of the text file.
    """
    with open(path, 'r') as file:
        return file.read()

def save_textfile(file_path: str, content: str) -> None:
    """
    Save the given content to a text file at the specified path.

    Args:
        file_path (str): Path where the text file will be saved.
        content (str): Content to write into the text file.
    """
    with open(file_path, 'w') as file:
        file.write(content)

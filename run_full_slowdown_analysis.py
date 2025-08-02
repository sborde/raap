"""
This script runs a full slowdown analysis on each cell of the dataset.

It iterates over the given input cells and for each cell:
 - calculates the slowdown features,
 - visualizes the distribution of the spike features
 - plots the spikes grouped by the features according to a given threshold
 - estimates and plots the SNR using the provided Savitzky-Golay filter

Versions:
   - 2025. 07. 19. - initial version
   - 2025. 07. 29. - way of starting the analysis changed
"""

import json
from typing import Callable
from pathlib import Path

from analyses.slowdown_analysis import analyze_cell
from common.run_utils import collect_inputs
from utils.report_generators import compose_textfile

def generate_report(
    figure_dicts: list[dict],
    text_composer: Callable[[str], None]
) -> None:
    """
    Generate a report for the analysis result figures as a HTML file.
    """
    json_string = json.dumps(
        figure_dicts,
        indent=4,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ': ')
    )
    text_composer(json_string)

def run_slowdown_analysis(config: dict) -> None:
    """
    Runs the full slowdown analysis on the dataset 
    based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary for the analysis.
    """

    template_path = Path('html_templates') / 'index.html'
    report_path = Path(config['output']['path']) / 'index_pooled.html'

    def report_composer(json_string: str) -> None:
        """
        Generates a report from the JSON string.
        """
        replacements = {'result_list': json_string}
        compose_textfile(template_path, replacements, report_path)

    def report_generator(figure_dicts: list[dict]) -> None:
        """
        Generates a report from the figure dictionaries.
        """
        generate_report(figure_dicts, report_composer)

    cells = collect_inputs(config)
    results = []
    for i, cell in enumerate(cells):
        print(f"Processing file {i+1}/{len(cells)}: {cell.name}")
        results.append(analyze_cell(cell, config))
        report_generator(results)

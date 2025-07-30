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
from pathlib import Path

from analyses.slowdown_analysis import analyze_cell
from utils.file_utils import enumerate_input_files

def run_slowdown_analysis(config: dict) -> None:
    """
    Runs the full slowdown analysis on the dataset 
    based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary for the analysis.
    """
    cell_paths = enumerate_input_files(config)

    cell_results = []
    for i, cell_path in enumerate(cell_paths):
        print(f"Processing file {i+1}/{len(cell_paths)}: {cell_path.name}")
        cell_results.append(analyze_cell(cell_path, config))

        json_string = json.dumps(
            cell_results,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ': ')
        )

        # Create or update the HTML browser.
        result_root = Path(config['output']['path'])
        html_path = result_root / 'index.html'
        with open('html_templates/index.html', 'r') as template_file:
            html_content = template_file.read()
        html_content = html_content.replace('[[cells_json]]', json_string)
        with open(html_path, 'w') as html_file:
            html_file.write(html_content)

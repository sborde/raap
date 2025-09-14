"""Module for HTML report generation utility functions."""

import json
from pathlib import Path

from .file_io import load_textfile, save_textfile


class TemplateFiller:
    """Class for filling JSON templates with data."""

    def __init__(self, template_path: str, output_path: str):
        self._template_path = template_path
        self._output_path = output_path

    def __call__(self, replacements: dict) -> None:
        """
        Fill the template with the provided replacements.
        """
        final_text = load_textfile(self._template_path)
        for placeholder, value in replacements.items():
            final_text = final_text.replace(f'[[{placeholder}]]', value)
        save_textfile(self._output_path, final_text)


class SimpleResultReportGenerator:
    """Class for generating reports from figure dictionaries."""

    def __init__(self, template_filler: TemplateFiller):
        self._template_filler = template_filler

    def __call__(self, figure_dicts: list[dict]) -> None:
        """
        Generate a report from the figure dictionaries.
        """
        json_string = json.dumps(
            figure_dicts,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ': ')
        )
        self._template_filler({'result_list': json_string})

def create_report_generator(config: dict) -> SimpleResultReportGenerator:
    """
    Create a SimpleResultReportGenerator based on the provided configuration.
    """
    template_path = f'html_templates/{config.get("template_name", "index")}.html'
    report_path = (
        Path(config['data']['features']['path']) /
        '02_figures' /
        'index.html'
    ).as_posix()
    return SimpleResultReportGenerator(
        TemplateFiller(template_path, report_path)
    )

# def collect_group_results(
#     group_path: Path,
#     group_name: str,
#     reference_path: Path
# ) -> list[dict]:
#     """
#     Traverses the given directory and returns a list of all files.
    
#     Args:
#         directory (Path): The directory to traverse.
        
#     Returns:
#         list[Path]: A list of file paths in the directory.
#     """
#     results = {}
#     for content in group_path.iterdir():
#         if not content.is_file():
#             continue
#         cell_name = '_'.join(content.stem.split('_')[:-1])
#         figure_type = content.stem.split('_')[-1]
#         if cell_name not in results:
#             results[cell_name] = []
#         results[cell_name].append(
#             (FIGURE_WEIGHT[figure_type], content.relative_to(reference_path))
#         )

#     return [
#         {
#             'name': cell_name,
#             'type': group_name,
#             'images': [
#                 file_path.as_posix() for _, file_path in 
#                 sorted(results[cell_name], key=lambda x: x[0])
#             ]
#         } for cell_name in sorted(results.keys())
#     ]

# def collect_grouped_results(
#     result_root: Path,
#     reference_path: Path
# ) -> list[dict]:
#     """
#     Traverses the given directory and returns a list of all files.
    
#     Args:
#         directory (Path): The directory to traverse.
#         reference_path (Path): The reference path to use for relative paths in results.
#     Returns:
#         list[Path]: A list of file paths in the directory.
#     """
#     groups = []
#     for entry in result_root.iterdir():
#         if not entry.is_dir():
#             continue
#         group_name = entry.name
#         group_results = collect_group_results(entry, entry.name, reference_path)
#         if group_results:
#             groups.append({
#                 'name': group_name,
#                 'results': group_results
#             })
#     return sorted(groups, key=lambda x: x['name'])
"""Module for HTML report generation utility functions."""

from pathlib import Path


def load_textfile(path: Path) -> str:
    """
    Load a text file from the specified path.
    
    Args:
        path (Path): Path to the text file.

    Returns:
        str: Content of the text file.
    """
    with open(path, 'r') as file:
        return file.read()

def save_textfile(file_path: Path, content: str) -> None:
    """
    Save the given content to a text file at the specified path.

    Args:
        file_path (Path): Path where the text file will be saved.
        content (str): Content to write into the text file.
    """
    with open(file_path, 'w') as file:
        file.write(content)

def fill_template(template: str, replacements: dict) -> str:
    """
    Replace placeholders in the template with specified values from 
    the replacements dictionary.
    
    Args:
        template (str): The template string containing placeholders.
        replacements (dict): A dictionary of placeholders and their replacements.
    
    Returns:
        str: The template with placeholders replaced by their corresponding values.
    """
    for placeholder, value in replacements.items():
        template = template.replace(f'[[{placeholder}]]', value)
    return template

def compose_textfile(
    template_path: Path,
    replacements: dict,
    output_path: Path
) -> None:
    """
    Generate a text file by replacing placeholders in the template with 
    specified values from the replacements dictionary and saves the result
    to the output path.
    
    Args:
        template_path (Path): The path to the text template file.
        replacements (dict): A dictionary of placeholders and their replacements.
        output_path (Path): The path where the generated text file will be saved.
    """
    template = load_textfile(template_path)
    final_text = fill_template(template, replacements)
    save_textfile(output_path, final_text)

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
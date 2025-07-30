

import json
from pathlib import Path

FIGURE_ORDER = [
    'residuals',
    'scatter',
    'piechart',
    'averages'
]

FIGURE_WEIGHT = {name: i for i, name in enumerate(FIGURE_ORDER)}

def get_html_template(template_name: str) -> str:
    """
    Returns the content of the HTML template file.
    
    Args:
        template_name (str): The name of the HTML template file.
        
    Returns:
        str: The content of the HTML template.
    """
    template_path = (
        Path(__file__).parent /
        'html_templates' /
        f'{template_name}.html'
    )
    with open(template_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_html_file(file_path: Path, content: str) -> None:
    """
    Writes the given content to an HTML file.
    
    Args:
        file_path (Path): The path where the HTML file will be saved.
        content (str): The content to write into the HTML file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def collect_group_results(
    path: Path,
    group_name: str,
    reference_path: Path
) -> list[dict]:
    """
    Traverses the given directory and returns a list of all files.
    
    Args:
        directory (Path): The directory to traverse.
        
    Returns:
        list[Path]: A list of file paths in the directory.
    """
    cell_results = {}
    for content in path.iterdir():
        if not content.is_file():
            continue
        cell_name = '_'.join(content.stem.split('_')[:-1])
        figure_type = content.stem.split('_')[-1]
        if cell_name not in cell_results:
            cell_results[cell_name] = []
        cell_results[cell_name].append(
            (FIGURE_WEIGHT[figure_type], content.relative_to(reference_path))
        )

    return [
        {
            'name': cell_name,
            'type': group_name,
            'images': [
                file_path.as_posix() for _, file_path in 
                sorted(cell_results[cell_name], key=lambda x: x[0])
            ]
        } for cell_name in sorted(cell_results.keys())
    ]

def collect_grouped_results(directory: Path, reference_path: Path) -> list[dict]:
    """
    Traverses the given directory and returns a list of all files.
    
    Args:
        directory (Path): The directory to traverse.
        
    Returns:
        list[Path]: A list of file paths in the directory.
    """
    groups = []
    for entry in directory.iterdir():
        if not entry.is_dir():
            continue
        group_name = entry.name
        group_results = collect_group_results(entry, entry.name, reference_path)
        if group_results:
            groups.append({
                'name': group_name,
                'results': group_results
            })
    return sorted(groups, key=lambda x: x['name'])



RESULTS_ROOT = Path(
    'C:/tmp/Bozso2025/PutRAAP/Results/IndividualSlowdownAnalysis'
)
FIGURE_DIR = RESULTS_ROOT / 'Figures'

groups = collect_grouped_results(FIGURE_DIR, RESULTS_ROOT)

json_string = json.dumps(
    groups,
    indent=4,
    sort_keys=True,
    ensure_ascii=False,
    separators=(',', ': ')
)

html_template = get_html_template('index_new')
html_output = html_template.replace('[[result_list]]', json_string)
html_path = RESULTS_ROOT / 'index_grouped.html'
write_html_file(html_path, html_output)

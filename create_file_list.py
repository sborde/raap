
from pathlib import Path

import pandas as pd

from run_peritrigger_event_analysis import CONFIG_NAME
from triggered_firing.io_utils import list_files_recursively

PARAMS = {
    'path': 'C:\\Datasets\\raap\\InVivoJuxta',
    'extension': 'mat'
}

OUTPUT_DIR = 'C:\\tmp\\Bozso2025\\PutRAAP\\Results\\TriggeredFiringComparison'

file_list = list_files_recursively(PARAMS['path'], PARAMS['extension'])
cell_list = []
for file in file_list:
    cell_list.append({
        'cell_type': file.parent.name,
        'name': file.stem,
        'group': 0
    })

cell_list_frame = pd.DataFrame(cell_list)
cell_list_frame.to_excel(OUTPUT_DIR + '\\cell_list.xlsx', index=False)
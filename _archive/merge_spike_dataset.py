"""
This script merges cellwise CSV files to a combined sheet
containing all spikes with their cell name and spike type.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path('C:/Datasets/raap/InVitroJuxtaSorted/20250625')
INPUT_DIR = BASE_DIR / 'Results'

data_frames = []
for file in INPUT_DIR.glob('*.csv'):
    print(f'Processing {file.name}')
    cell_name = '_'.join(file.stem.split('_')[:2])
    spike_type = file.stem.split('_')[2].replace('juxta', '')
    df = pd.read_csv(file)
    df['cell_name'] = cell_name
    df['spike_type'] = spike_type
    data_frames.append(df)

dataset = pd.concat(data_frames, ignore_index=True)
dataset.to_csv(BASE_DIR / 'invitro_juxta_dataset.csv', index=False)
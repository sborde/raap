
from pathlib import Path
import pandas as pd

INPUT_PATH = (
    Path('E:/Data/BogiSejtek/PutRAAP/Results') /
    '20250702' /
    'InVitroJuxta' /
    'DvMaxNorm' /
    'Search_0.15-0.85' /
    'w11_p2' /
    'Preprocessed'
)

skip_cells = ['2407032SI_GR1']

datas = []
for file in INPUT_PATH.glob('*.csv'):
    print(f'Processing file: {file.name}')
    df = pd.read_csv(file)
    if df['cell_name'].iloc[0] in skip_cells:
        print(f'Skipping cell: {df["cell_name"].iloc[0]}')
        continue
    datas.append(df)

pool_data = pd.concat(datas, ignore_index=True)

pool_data.to_csv(
    INPUT_PATH.parent / 'invitro_juxta_dataset.csv',
    index=False
)

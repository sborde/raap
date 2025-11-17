"""A module to hold functions and classes for spike type predictions."""

from pathlib import Path
from typing import Optional

from tqdm import tqdm

import numpy as np
import pandas as pd

from common.data_utils import CellWrapper
from workflows.run_util.io import ResultPaths

def collect_predictive_features(
    input_files: list[CellWrapper[ResultPaths]],
    type_mapper: dict[str, int],
    episode_names: list[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Collect predictive features from input files for spike type prediction.
    """
    property_frames = []
    for input_file in tqdm(
        input_files,
        desc='Collect features',
        total=len(input_files)
    ):
        input_data = input_file.data.load()
        cells_properties = []
        for _, spike in input_data.spikes.iterrows():
            if spike['spike_type'] == 'UNKNOWN':
                continue
            spike_time = spike['timestamp']
            properties = {
                'y': type_mapper[spike['spike_type']],
                'cell_name': input_file.name,
                'cell_type': input_file.cell_type
            }
            for episode_name in episode_names:
                try:
                    starts, ends = input_data.get_episode_ends(episode_name)
                    starts, ends = starts[starts <= spike_time], ends[ends <= spike_time]
                    if len(starts) == 0 or len(ends) == 0:
                        continue
                    properties[f'{episode_name}_start_dt'] = spike_time - np.max(starts)
                    properties[f'{episode_name}_end_dt'] = spike_time - np.max(ends)
                except KeyError:
                    continue
            cells_properties.append(properties)
        cell_df = pd.DataFrame(cells_properties)
        if save_path is not None:
            save_dir = Path(save_path) / '01_results' / input_file.cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            cell_df.to_csv(
                save_dir / f'{input_file.name}_predictive_features.csv',
                index=False
            )
        property_frames.append(cell_df)
    return pd.concat(property_frames, ignore_index=True)
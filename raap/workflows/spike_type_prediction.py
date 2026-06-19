"""A module to hold functions and classes for spike type predictions."""

from pathlib import Path
from typing import Optional

from tqdm import tqdm

import numpy as np
import pandas as pd

from common.data_utils import CellWrapper
from workflows.run_util.io import ResultPaths

def collect_spike_pattern(
    input_files: list[CellWrapper[ResultPaths]],
    type_mapper: dict[str, int],
    n_preceeding: int,
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
        times = input_data.spikes['timestamp'].values
        spike_types = input_data.spikes['spike_type'].values
        spike_props = []
        for idx in range(n_preceeding, len(times)):
            current_type = spike_types[idx]
            if current_type not in type_mapper:
                continue
            preceding_types = spike_types[idx - n_preceeding: idx]
            preceding_times = times[idx] - times[idx - n_preceeding: idx]
            features = {'y': type_mapper[current_type]}
            for i in range(n_preceeding):
                features[f'raap_-{n_preceeding-i}'] = (
                    1 if preceding_types[i] == 'RA-AP' else 0
                )
                features[f'dt_-{n_preceeding-i}'] = preceding_times[i]
            spike_props.append(features)
        cell_frame = pd.DataFrame(spike_props)
        cell_frame.insert(0, 'cell_name', input_file.name)
        cell_frame.insert(1, 'cell_type', input_file.cell_type)
        if save_path is not None:
            save_dir = Path(save_path) / '01_results' / input_file.cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            cell_frame.to_csv(
                save_dir / f'{input_file.name}_spike_pattern_features.csv',
                index=False
            )
        property_frames.append(cell_frame)
    return pd.concat(property_frames, ignore_index=True)



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
        cell_frame = input_data.spikes[['timestamp', 'spike_type']].copy()
        cell_frame.insert(0, 'cell_name', input_file.name)
        cell_frame.insert(1, 'cell_type', input_file.cell_type)
        cell_frame['y'] = cell_frame['spike_type'].map(type_mapper)
        for episode_name in episode_names:
            try:
                starts, ends = input_data.get_episode_ends(episode_name)
                for trigger_name, trigger_times in [
                    ('start', starts), ('end', ends)
                ]:
                    for trigger in trigger_times:
                        after_flag = cell_frame['timestamp'] >= trigger
                        cell_frame.loc[
                            after_flag,
                            f'{episode_name}_{trigger_name}_dt'
                        ] = cell_frame.loc[after_flag, 'timestamp'] - trigger
            except KeyError:
                continue
        if save_path is not None:
            save_dir = Path(save_path) / '01_results' / input_file.cell_type
            save_dir.mkdir(parents=True, exist_ok=True)
            cell_frame.to_csv(
                save_dir / f'{input_file.name}_predictive_features.csv',
                index=False
            )
        property_frames.append(cell_frame)
    return pd.concat(property_frames, ignore_index=True)

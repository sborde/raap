"""This module contains functions and classes for noise threshold decision."""

import numpy as np
import pandas as pd

from analyses.model import RecordingWithSpikes
from common.data_utils import CellWrapper
from workflows.run_util.io import ResultPaths

def collect_pooled_data(
    input_files: list[CellWrapper[ResultPaths]]
) -> pd.DataFrame:
    """Collects noise spike features for all cells."""
    return pd.concat([
        collect_noise_data(input_file)
        for input_file in input_files
    ], ignore_index=True)

def collect_aggregated_data(
    input_files: list[CellWrapper[ResultPaths]]
) -> pd.DataFrame:
    """Pools noise spike features to calculate noise properties."""
    return pd.concat([
        prepare_cell_noise_data(input_file)
        for input_file in input_files
    ], ignore_index=True)

def prepare_cell_noise_data(
    input_file: CellWrapper[ResultPaths]
) -> pd.DataFrame:
    """Prepares noise spike features for each cell."""
    data = input_file.data.load()
    nrmsd = calculate_nrmsd(data)
    stats = calculate_descriptive_stats(nrmsd)
    df = pd.DataFrame({
        'cell_name': input_file.info.name,
        'cell_type': input_file.info.cell_type,
    } | stats, index=[0])
    return df

def collect_noise_data(
    input_file: CellWrapper[ResultPaths]
) -> pd.DataFrame:
    """Collects noise spike features for each cell."""
    data = input_file.data.load()
    nrmsd = calculate_nrmsd(data)
    return pd.DataFrame({
        'cell_name': input_file.info.name,
        'cell_type': input_file.info.cell_type,
        'nrmsd': nrmsd
    })

def calculate_nrmsd(
    data: RecordingWithSpikes
) -> np.ndarray:
    """Collects noise spike features for each cell."""
    rmsd = np.sqrt(data.spikes['noise_power'])
    amplitude = data.spikes['amplitude']
    return rmsd / amplitude

def calculate_descriptive_stats(
    data: np.ndarray
) -> dict:
    """Aggregates noise spike features to cell-wise statistics."""
    return {
        'min': data.min(),
        'max': data.max(),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'median': np.percentile(data, 50),
        'mean': np.mean(data),
    }


from pathlib import Path
from typing import Callable

import pandas as pd

from common.data_utils import CellWrapper

from .calculators import SpikeTypeResults

def convert_spike_result_to_df(
    result: SpikeTypeResults
) -> pd.DataFrame:
    """Convert SpikeTypeResults to a DataFrame."""
    histogram = result.result
    result_dict = {
        'centers': histogram.centers,
        'spike_type': result.spike_type
    }
    for i in range(histogram.n_trials):
        result_dict[f'trigger_{i}'] = histogram.values[:, i]
    return pd.DataFrame(result_dict)

def export_spike_result_to_csv(
    cell_results: CellWrapper[list[SpikeTypeResults]],
    base_path: str,
    suffix: str
):
    """Export SpikeTypeResults of a cell as a CSV."""
    info = cell_results.info
    results = cell_results.data

    results_df = pd.concat([
        convert_spike_result_to_df(result) for result in results
    ])

    cell_output_path = (
        Path(base_path) /
        'statistics' /
        info.cell_type
    )
    cell_output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        cell_output_path / f'{info.name}_{suffix}_histogram.csv',
        index=False
    )

def create_result_saver(path: str, suffix: str) -> Callable[[CellWrapper[list[SpikeTypeResults]]], None]:
    def save(cell_result: CellWrapper[list[SpikeTypeResults]]) -> None:
        export_spike_result_to_csv(cell_result, path, suffix)
    return save

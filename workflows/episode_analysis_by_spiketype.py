

from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

from common.data_utils import CellInfo, CellWrapper
from analyses.model import (
    RecordingWithSpikes,
    mark_events_around_trigger,
    mark_events_in_interval
)
from workflows.run_util.grouping import get_grouping
from workflows.run_util.io import (
    ResultPaths,
    RecordingWithSpikes,
    setup_result_dir
)
from workflows.run_util.plotting import CategoricalDistribution, create_stacked_area_chart


class EpisodeFiringBySpikeType:
    """
    An analysis class for the analysis of the relationship 
    between spike types and sleep episodes or sections.
    """

    def __init__(self, config: dict):
        self._output_dirs = setup_result_dir(config['output']['path'])
        self._episodes = config['params']['episodes']
        self._spike_types = config['params']['spike_types']
        self._plot_params = config['plot']
        self._palette = config['palette']
        if config['grouping']['do']:
            self._grouping = pd.read_excel(config['grouping']['path'])

    def run(self, cell: CellWrapper[ResultPaths]) -> CellWrapper[DataFrame]:
        """
        Run the spike type distribution analysis.
        """
        # Load data
        data = cell.data.load()
        # Calculate firing rate during episodes
        result_list = []
        for episode in self._episodes:
            try:
                result_list.append(self._process_episode(data, episode))
            except KeyError:
                print(
                    f"[INFO] Episode '{episode}' is missing in {cell.info.name}."
                )
                continue
        result = pd.concat(result_list, ignore_index=True)

        result_dir = self._output_dirs['figures'] / cell.info.cell_type
        result_dir.mkdir(parents=True, exist_ok=True)
        result.to_excel(
            result_dir /
            f"{cell.info.name}_episode_firing_rate_by_spiketype.xlsx",
            index=False
        )
        return CellWrapper(cell.info, result)

    def _process_episode(self, data: RecordingWithSpikes, name: str) -> DataFrame:
        """
        Calculates the mean firing rate of each spike type
        during each interval of the given type of episode.
        """
        spikes = data.spikes.copy()
        timestamps = spikes['timestamp'].values
        starts, ends = data.get_episode_ends(name)
        for start, end in zip(starts, ends):
            duration = end - start
            is_within = (timestamps >= start) & (timestamps <= end)
            result = {
                'episode': name,
                'start': start,
                'end': end,
                'duration': duration
            }
            for spike_type in self._spike_types:
                result[spike_type] = np.sum(
                    is_within & (spikes['spike_type'] == spike_type)
                ) / duration
        return DataFrame(result, index=[0])

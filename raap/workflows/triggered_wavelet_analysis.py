"""This module contains classes and function to run spike autocorrelation analysis workflows."""


from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pandas import DataFrame

from analyses.model import (
    RecordingWithSpikes,
    select_spikes
)
from common.data_utils import CellWrapper
from param_classes.workflows.wavelet_analysis import SpikeTriggeredWaveletWorkflowParams

from notebook_utils.recording_wavelet import plot_spectrogram
from raap.workflows.perform_full_wavelet import calculate_spectrogram, Spectrogram

from workflows.run_util.io import ResultPaths
from workflows.spike_ac_analysis import (
    save_figure,
)


def run_analysis(
    params: SpikeTriggeredWaveletWorkflowParams,
    input_files: list[CellWrapper[ResultPaths]]
) -> list[CellWrapper[DataFrame]]:
    """Run the spike autocorrelation analysis workflow on the given input files."""
    analysis = SpikeTriggeredWaveletAnalysis(params)
    return [analysis(input_file) for input_file in input_files]


class SpikeTriggeredWaveletAnalysis:
    """Class to perform spike autocorrelation analysis for individual spike types."""
    def __init__(self, config: SpikeTriggeredWaveletWorkflowParams):
        self._params = config.analysis
        self._spike_types = self._params.spike_types

        self._cut_around = partial(
            Spectrogram.cut_around,
            window=self._params.cut_window
        )

        self._to_svg = self._params.to_svg
        self._palette = config.palette

        self._ylim = self._params.ylim
        self._xlim = self._params.xlim
        self._clim = self._params.clim
        self._to_svg = self._params.to_svg
        self._to_png = self._params.to_png
        self._dpi = self._params.dpi

        self._paths = None
        if config.output.path:
            self._paths = {
                name: Path(config.output.path) / dir_name
                for name, dir_name in zip(
                    ['results', 'figures'], ['01_results', '02_figures']
                )
            }
            for path in self._paths.values():
                path.mkdir(parents=True, exist_ok=True)

    def __call__(self, cell: CellWrapper[ResultPaths]) -> CellWrapper:
        """Process a single input file and return the results."""
        cell_data = cell.data.load()
        spectrogram = self._calculate_spectrogram(cell_data)
        spectrogram_segments = self._segment_by_spike_type(spectrogram, cell_data.spikes)
        cell_result = CellWrapper(cell.info, spectrogram_segments)
        self._plot(cell_result)
        # if self._paths and self._paths.get('results'):
        #     save_spectrogram_hdf(
        #         spectrogram,
        #         cell.name,
        #         str(self._paths['results'])
        #     )
        self._save_avg_spectrum(cell_result)
        return cell_result

    def _calculate_spectrogram(
        self, cell_data: RecordingWithSpikes
    ) -> Spectrogram:
        """Calculate the spectrogram of a signal."""
        signal = cell_data.get_signal(**self._params.signal.model_dump())
        return calculate_spectrogram(signal, self._params.wavelet)

    def _segment_by_spike_type(
        self, spectrogram: Spectrogram, spikes: DataFrame
    ) -> dict[str, list[Spectrogram]]:
        """Calculate the normalized inter RA-AP interval firing histogram."""
        return {
            spike_type: [
                sub_spec for trigger in select_spikes(spikes, spike_type)
                if (sub_spec := self._cut_around(spectrogram, trigger)) is not None
            ] for spike_type in self._spike_types
        }

    def _aggregate(
        self, spectrograms: list[Spectrogram]
    ) -> Spectrogram:
        """Average multiple spectrograms into a single spectrogram."""
        avg_power = np.mean(
            np.stack([spec.power for spec in spectrograms], axis=0), axis=0
        )
        return Spectrogram(
            spectrograms[0].times,
            spectrograms[0].freqs,
            avg_power,
            spectrograms[0].fs
        )

    def _plot(
        self,
        cell_result: CellWrapper[dict[str, list[Spectrogram]]]
    ):
        """Plot the results."""
        cell_name = cell_result.info.name
        spike_colors = self._palette['spike_type'] | {
            'AIS-AP': self._palette['cell_type'][cell_result.info.cell_type]
        }

        for trigger_name, trigger_results in cell_result.data.items():
            fig = plt.figure(
                figsize=(10, 6), constrained_layout=True
            )
            grids = fig.add_gridspec(
                nrows=1,
                ncols=2,
                width_ratios=[9, 1]
            )
            spec_ax = fig.add_subplot(grids[0, 0])
            power_ax = fig.add_subplot(grids[0, 1])

            avg_spectrogram = self._aggregate(trigger_results)
            plot_spectrogram(
                avg_spectrogram,
                clim=self._params.clim,
                fig=fig,
                ax=spec_ax
            )

            spectra = np.stack([
                np.mean(spec.power, axis=1) for spec in trigger_results
            ], axis=1)
            mean_spectrum = np.mean(spectra, axis=1)
            std_spectrum = np.std(spectra, axis=1)
            sem_spectrum = std_spectrum / np.sqrt(spectra.shape[1])
            power_ax.fill_betweenx(
                avg_spectrogram.freqs,
                mean_spectrum - sem_spectrum,
                mean_spectrum + sem_spectrum,
                color=spike_colors[trigger_name],
                edgecolor='none',
                alpha=0.3
            )
            power_ax.plot(
                mean_spectrum,
                avg_spectrogram.freqs,
                '-',
                linewidth=2,
                color=spike_colors[trigger_name],
            )

            spec_ax.set_ylim(self._ylim)
            spec_ax.set_xlim(self._xlim)
            power_ax.set_ylim(self._ylim)

            plt.suptitle(f'{cell_name} - {trigger_name}')

            if self._paths:
                save_figure(
                    self._paths['figures'],
                    fig,
                    (
                        f'{cell_name}_' +
                        f'{trigger_name}_power'
                    ),
                    self._to_svg
                )
            else:
                plt.show()

    def _save_avg_spectrum(
        self,
        cell_result: CellWrapper[dict[str, list[Spectrogram]]]
    ):
        """Save the results to file."""
        if not self._paths:
            return

        cell_name = cell_result.info.name
        frames = []
        for spike_type, spectrograms in cell_result.data.items():
            col_names = [f'{f}_Hz' for f in spectrograms[0].freqs]
            spectra = np.stack([
                    np.mean(spec.power, axis=1)
                    for spec in spectrograms
                ], axis=0
            )
            df = DataFrame(spectra, columns=col_names)
            df.insert(0, 'spike_type', spike_type)
            frames.append(df)

        save_name = f'{cell_name}_spike_freq_spectrum'

        pd.concat(frames, ignore_index=True).to_excel(
            f'{self._paths["results"]}/{save_name}.xlsx',
            index=False
        )

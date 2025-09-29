

from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

from analyses.slowdown_descriptors import find_start_index
from common.data_utils import CellWrapper
from data.spikes import SpikeShape, SpikeShapes, load_spike_shapes
from triggered_firing.io_utils import load_input_data
from .dto.spike_shapes import PreparedSpikeShape

class BasicSpikeFeatureDescription:
    """
    A basic spike feature description class.
    """

    def __init__(self, config: dict):
        self._config = config
        self._suffix = config.get('suffix', 'basic_features')
        self._params = config['params']
        self._result_dir = Path(config['output']['path'])

    def __call__(self, input_file: CellWrapper[dict[str, Path]]):
        # input_data = load_input_data(input_file)
        spikes = self._extract_spikes(input_file)
        features = pd.DataFrame(self._describe_spikes(spikes))
        output_path = (
            self._result_dir /
            input_file.info.cell_type /
            f"{input_file.info.name}_{self._suffix}.xlsx"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_excel(output_path, index=False)

    @property
    def suffix(self) -> str:
        """The suffix for the output files."""
        return self._suffix

    def _extract_spikes(self, input_file: CellWrapper[dict[str, Path]]):
        # Load spikes from the input file
        return load_spike_shapes(input_file.data['raw'], self._params['extractor'])

    def _describe_spikes(self, spikes: SpikeShapes):
        # Describe the spikes and extract features
        features = list(filter(
            lambda x: x is not None,
            [self._describe_spike(spike) for spike in spikes]
        ))
        if len(features) < len(spikes):
            n_errors = len(spikes) - len(features)
            print(f"[INFO]: {n_errors} spike's start couldn't be detected.")
        return features

    def _describe_spike(self, spike: SpikeShape) -> Optional[dict]:
        # Describe a single spike and extract features
        prepared_spike = self._prepare_spike(spike)
        if prepared_spike is None:
            return None
        amplitude = self._calculate_amplitude(prepared_spike)
        half_left, half_right = self._calculate_halfwidth(prepared_spike)
        rise_time = self._calculate_risetime(prepared_spike)
        rising_slope = amplitude / rise_time
        base_v = self._calculate_base_v(prepared_spike)
        prespike_slope, prespike_intercept = self._calculate_prespike_slope(prepared_spike)

        # plt.figure()
        # plt.plot(prepared_spike.spike.times, prepared_spike.spike.values, '-')
        # plt.plot(
        #     prepared_spike.spike.times[prepared_spike.start],
        #     prepared_spike.spike.values[prepared_spike.start],
        #     'ro'
        # )
        # plt.plot(
        #     prepared_spike.spike.times[prepared_spike.peak],
        #     prepared_spike.spike.values[prepared_spike.peak],
        #     'go'
        # )
        # plt.plot(
        #     [half_left[0], half_right[0]],
        #     [half_left[1], half_right[1]],
        #     'k--',
        #     label='Halfwidth'
        # )
        # plt.plot(
        #     [prepared_spike.spike.times[prepared_spike.start], prepared_spike.spike.times[prepared_spike.peak]],
        #     [prepared_spike.spike.values[prepared_spike.start], prepared_spike.spike.values[prepared_spike.start]],
        #     'm--',
        #     label='Rise time'
        # )
        # plt.plot(
        #     [prepared_spike.spike.times[prepared_spike.peak], prepared_spike.spike.times[prepared_spike.peak]],
        #     [prepared_spike.spike.values[prepared_spike.start], prepared_spike.spike.values[prepared_spike.peak]],
        #     'g--',
        #     label='Amplitude'
        # )
        # plt.plot(
        #     [prepared_spike.spike.times[prepared_spike.start], prepared_spike.spike.times[prepared_spike.peak]],
        #     [prepared_spike.spike.values[prepared_spike.start], prepared_spike.spike.values[prepared_spike.peak]],
        #     'b--',
        #     label='Rising slope'
        # )
        # plt.plot(
        #     prepared_spike.spike.times[:prepared_spike.start+1],
        #     prespike_slope * prepared_spike.spike.times[:prepared_spike.start+1] + prespike_intercept,
        #     'y--',
        #     label='Prespike slope'
        # )
        # plt.axhline(base_v, color='c', linestyle='--', label='Base V')
        # plt.legend(loc='upper right')
        # plt.show()

        return {
            'timestamp': spike.timestamp,
            'rmsd': prepared_spike.noise_properties['rmsd'],
            'amplitude': amplitude,
            'rise_time': rise_time * 1000,
            'rising_slope': rising_slope,
            'halfwidth': (half_right[0] - half_left[0]) * 1000,
            'start_v': prepared_spike.spike.values[prepared_spike.start],
            'base_v': base_v,
            'prespike_slope': prespike_slope,
        }

    def _prepare_spike(self, spike: SpikeShape) -> Optional[PreparedSpikeShape]:
        # Prepare spike for feature extraction

        peak_i = np.argmin(np.abs(spike.times - 0)).astype(int)
        start_i = find_start_index(
            spike.values[:peak_i + 1], self._params['descriptor']
        )

        # if the start cannot be detected, return None
        if start_i is None:
            return None

        spike, noise_properties = self._smooth_spike(spike)
        peak_i = int(np.argmin(np.abs(spike.times - 0)).astype(int))

        return PreparedSpikeShape(spike, start_i, peak_i, noise_properties)

    def _smooth_spike(self, spike: SpikeShape) -> tuple[SpikeShape, Optional[dict]]:
        # Smooth the spike and return the smoothed spike and noise properties
        smooth_params = self._params['descriptor'].get('smooth', None)
        if smooth_params is None:
            return spike, None
        v_range = np.max(spike.values) - np.min(spike.values)
        v_smooth = savgol_filter(spike.values, **smooth_params)
        residual_power = np.mean((spike.values - v_smooth) ** 2)
        noise_properties = {
            'residual_power': residual_power,
            'rmsd': np.sqrt(residual_power),
            'nrmsd': np.sqrt(residual_power) / v_range
        }
        return (
            SpikeShape(spike.times, v_smooth, spike.timestamp, spike.dt),
            noise_properties
        )

    def _calculate_amplitude(self, prepared_spike: PreparedSpikeShape) -> float:
        """Calculate the amplitude of the spike."""
        return (
            prepared_spike.spike.values[prepared_spike.peak] -
            prepared_spike.spike.values[prepared_spike.start]
        )

    def _calculate_risetime(self, prepared_spike: PreparedSpikeShape) -> float:
        # Calculate the rise time of the spike
        spike = prepared_spike.spike
        start_t = spike.times[prepared_spike.start]
        peak_t = spike.times[prepared_spike.peak]
        return peak_t - start_t

    def _calculate_base_v(self, prepared_spike: PreparedSpikeShape) -> float:
        # Calculate the base voltage of the spike
        return float(np.mean(prepared_spike.spike.values[:prepared_spike.start+1]))

    def _calculate_prespike_slope(
        self,
        prepared_spike: PreparedSpikeShape
    ) -> tuple[float, float]:
        # Calculate the prespike slope of the spike
        spike = prepared_spike.spike
        start_i = prepared_spike.start
        if start_i < 2:
            return 0.0, 0.0
        X = spike.times[:start_i+1].reshape(-1, 1)
        y = spike.values[:start_i+1]
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0]), float(model.intercept_)

    def _calculate_halfwidth(
        self,
        prepared_spike: PreparedSpikeShape
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        # Calculate the half-width of the spike
        spike_v = prepared_spike.spike.values
        amplitude = self._calculate_amplitude(prepared_spike)
        start_v = prepared_spike.spike.values[prepared_spike.start]
        half_amplitude = amplitude / 2 + start_v
        i = prepared_spike.start
        while i < prepared_spike.peak and spike_v[i + 1] < half_amplitude:
            i += 1
        alpha = (half_amplitude - spike_v[i]) / (spike_v[i + 1] - spike_v[i])
        left_time = prepared_spike.spike.times[i] + alpha * prepared_spike.spike.dt
        i = prepared_spike.peak
        while i < len(spike_v) - 1 and spike_v[i + 1] > half_amplitude:
            i += 1
        alpha = (half_amplitude - spike_v[i]) / (spike_v[i + 1] - spike_v[i])
        right_time = prepared_spike.spike.times[i] + alpha * prepared_spike.spike.dt
        return (
            (left_time, half_amplitude),
            (right_time, half_amplitude)
        )


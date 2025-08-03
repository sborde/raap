


import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from data.spike import create_start_finder
from data.spike.processors import SavGolFilter
from data.spikes import SpikeShape, SpikeShapes
from signalizer.model.experiment.recordings import MatRecording
from signalizer.model.experiment.sequences import EpisodeSequence





class RisingFeatureCalculator:
    """
    A class to calculate rising phase features of spikes.
    """

    def __init__(self, config: dict):
        if 'smooth_params' in config and config['smooth_params']:
            self._smoother = SavGolFilter(**config['smooth_params'])
        else:
            self._smoother = None
        self._config = config
        self._n_spikes = 0

    def __call__(self, spikes: SpikeShapes) -> pd.DataFrame:
        """
        Calculate the features for a list of spikes.
        """
        self._n_spikes = len(spikes)
        features = []
        for i, spike in enumerate(spikes):
            print(f'Processing spike {i + 1}/{self._n_spikes}...', end='\r')
            feature = self._calculate(spike)
            if feature is not None:
                features.append(feature)
        print()

        if len(features) < len(spikes):
            n_errors = len(spikes) - len(features)
            print(f'[INFO]: {n_errors} spike\'s start couldn\'t be detected.')

        return self._convert_to_dataframe(features)

    def _convert_to_dataframe(self, features: list[dict]) -> pd.DataFrame:
        """
        Convert the list of features to a pandas DataFrame.
        """

        rows = []
        for f in features:
            rows.append({
                'timestamp': f['timestamp'],
                'relative_dv_drop': f['relative_dv_drop'],
                'relative_rebound_time': f['relative_rebound_time'],
                'raw_power': f['raw_power'],
                'smoothed_power': f['smoothed_power'],
                'noise_power': f['noise_power'],
                'start_t': f['t'][f['start_i']],
                'start_v': f['v'][f['start_i']],
                'peak_t': f['t'][np.argmin(np.abs(f['t']))],
                'peak_v': f['v'][np.argmin(np.abs(f['t']))],
                'search_start_t': f['t'][f['search_start_i']],
                'search_start_v': f['v'][f['search_start_i']],
                'search_end_t': f['t'][f['search_end_i']],
                'search_end_v': f['v'][f['search_end_i']],
                'left_peak_t': f['t'][f['left_peak']] if f['left_peak'] is not None else None,
                'left_peak_v': f['v'][f['left_peak']] if f['left_peak'] is not None else None,
                'right_peak_t': f['t'][f['right_peak']] if f['right_peak'] is not None else None,
                'right_peak_v': f['v'][f['right_peak']] if f['right_peak'] is not None else None,
                'valley_t': f['t'][f['valley_i']] if f['valley_i'] is not None else None,
                'valley_v': f['v'][f['valley_i']] if f['valley_i'] is not None else None,
                'rebound_start_t': f['t'][f['rebound_i'][0]] if f['rebound_i'] else None,
                'rebound_end_t': f['t'][f['rebound_i'][1]] if f['rebound_i'] else None
            })

        return pd.DataFrame(rows)

    def _describe_noise(self, spike: SpikeShape) -> dict:
        """
        Calculate the noise properties of a spike.
        """
        if self._smoother is None:
            return {
                'raw_power': np.nan,
                'smoothed_power': np.nan,
                'noise_power': np.nan
            }
        raw_v = spike.values
        smoothed_v = self._smoother(spike).values
        return {
            'raw_power': np.mean(raw_v ** 2),
            'smoothed_power': np.mean(smoothed_v ** 2),
            'noise_power': np.mean((raw_v - smoothed_v) ** 2)
        }

    def _find_start(self, spike: SpikeShape) -> int | None:
        """
        Find the start index of the rising phase of a spike.
        """
        start_finder = create_start_finder(self._config)
        return start_finder(spike)

    def _find_rise_start_index(
        self,
        v: np.ndarray,
        start_i: int,
        peak_i: int,
        threshold: float
    ) -> int:
        """
        Find the start index of the rising phase of a spike.
        The rising phase is defined as the first point after the start index
        where the signal exceeds a certain proportion of its amplitude.
        """
        # align rising phase to the start
        rising_phase = v[start_i:peak_i + 1] - v[start_i]
        threshold_level = threshold * rising_phase[-1]

        # find the first index where the rising phase exceeds the threshold
        return np.where(rising_phase > threshold_level)[0][0] + start_i

    def _find_search_start(
        self,
        start_i: int,
        peak_i: int,
        v: np.ndarray,
    ):
        # else on the rising phase from the given proportion of the amplitude
        params = self._config['search_start']
        if params['mode'] == 'amplitude':
            return self._find_rise_start_index(
                v,
                start_i,
                peak_i,
                params['param']['threshold']
            )
        elif params['mode'] == 'time':
            rising_sample_count = peak_i - start_i
            return np.round(
                rising_sample_count * params['param']['threshold']
            ).astype(int) + start_i
        return start_i

    def _calculate(self, spike: SpikeShape) -> dict | None:
        """
        Calculate the features for a single spike.
        """
        params = self._config
        start_i = self._find_start(spike)

        # if the start cannot be detected, return None
        if start_i is None:
            return None

        noise_properties = self._describe_noise(spike)
        v = self._smoother(spike).values if self._smoother else spike.values

        peak_i = np.argmin(np.abs(spike.times - 0)).astype(int)

        # search on the rising phase by default
        search_start_i = self._find_search_start(
            start_i,
            peak_i,
            v
        )

        # Find the peaks of the first derivative
        dv = np.diff(v)
        # search until the peak by default
        search_end_i = peak_i
        # else until the global maximum of the first derivative
        if params['search_end']['mode'] == 'dv_max':
            search_end_i = np.argmax(dv)
        elif params['search_end']['mode'] == 'time':
            rising_sample_count = peak_i - start_i
            search_end_i = peak_i - np.round(
                rising_sample_count * params['search_end']['param']['threshold']
            ).astype(int)


        # Calculate the rising phase duration
        dt = np.mean(np.diff(spike.times))
        rising_duration = (search_end_i - search_start_i) * dt

        # find peaks and keep only the valid ones
        peaks, _ = find_peaks(dv, prominence=(None, None))
        peaks = [peak for peak in peaks if search_start_i <= peak <= search_end_i]

        # if there are no secondary peaks, return default values
        if len(peaks) < 2:
            return {
                'timestamp': spike.timestamp,
                'relative_dv_drop': 0,
                'relative_rebound_time': 0,
                't': spike.times,
                'v': v,
                'dv': dv,
                'start_i': start_i,
                'search_start_i': search_start_i,
                'search_end_i': search_end_i,
                'left_peak': None,
                'right_peak': None,
                'valley_i': None,
                'rebound_i': [],
                'rebound_time': None,
                'dv_min_peak': None
            } | noise_properties

        features = []
        dv_max_i = np.argmax(dv[:peak_i])
        for peak in peaks:
            if peak == dv_max_i:
                # skip the global maximum peak
                continue

            left_peak, right_peak = (
                (peak, dv_max_i) if peak < dv_max_i else (dv_max_i, peak)
            )
            dv_interpeak = dv[left_peak:right_peak + 1]
            dv_min_peak = min(dv_interpeak[0], dv_interpeak[-1])

            # Find the valley between the peaks
            dv_valley_i = np.argmin(dv_interpeak)
            dv_valley = dv_interpeak[dv_valley_i]

            # Calculate the relative drop in dv
            dv_drop = dv_min_peak - dv_valley

            # find the first point left of the valley where dv reaches the min peak
            left_level_i = np.where(dv_interpeak <= dv_min_peak)[0][0]
            # find the first point right of the valley where dv reaches the min peak
            right_level_i = np.where(dv_interpeak <= dv_min_peak)[0][-1]

            # calculate the real rebound time
            rebound_time = (right_level_i - left_level_i) * dt
            if dv_interpeak[left_level_i] < dv_min_peak:
                alpha = (
                    (dv_min_peak - dv_interpeak[left_level_i]) /
                    (dv_interpeak[left_level_i-1] - dv_interpeak[left_level_i])
                )
                rebound_time += alpha * dt
            if dv_interpeak[right_level_i] < dv_min_peak:
                alpha = (
                    (dv_min_peak - dv_interpeak[right_level_i]) /
                    (dv_interpeak[right_level_i+1] - dv_interpeak[right_level_i])
                )
                rebound_time += (1 - alpha) * dt

            features.append({
                'timestamp': spike.timestamp,
                'dv_drop': dv_drop,
                'relative_dv_drop': dv_drop / dv[dv_max_i],
                'relative_rebound_time': rebound_time / rising_duration,
                't': spike.times,
                'v': v,
                'dv': dv,
                'start_i': start_i,
                'search_start_i': search_start_i,
                'search_end_i': search_end_i,
                'left_peak': left_peak,
                'right_peak': right_peak,
                'valley_i': dv_valley_i + left_peak,
                'rebound_i': [left_level_i + left_peak, right_level_i + left_peak],
                'rebound_time': rebound_time,
                'dv_min_peak': dv_min_peak
            } | noise_properties)

        return sorted(features, key=lambda x: x['dv_drop'])[-1]


class SpikeCategorizer:
    """
    A class to categorize spikes based on their features.
    """

    def __init__(self, features: list[dict], groups: dict):
        self._features = features
        self._groups = groups

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Categorize spikes based on the provided configuration.
        Args:
            cell_features (CellWrapper[pd.DataFrame]): The features of the cell spikes.
            config (dict): Configuration dictionary containing categorization settings.
        Returns:
            CellWrapper[pd.DataFrame]: A DataFrame of spike features with category labels
        """
        data = data.copy()
        data['spike_type'] = self._groups[0]['name']  # Default category
        for group in self._groups[1:]:
            flags = []
            for feature, bottom in zip(self._features, group['bottom-left']):
                flags.append(bottom < data[feature['name']])
            flag = np.logical_and.reduce(flags)
            data.loc[flag, 'spike_type'] = group['name']
        data['spike_type'] = data['spike_type'].fillna('other')
        data['spike_type'] = pd.Categorical(
            data['spike_type'],
            categories=[g['name'] for g in self._groups]
        )
        return data


class SpikeIntervalAssignerFactory:
    """
    A factory class to create interval assigners based on episode names.
    """

    def __init__(self, episode_names: list[str]):
        self._episode_names = episode_names

    def __call__(self, recording: MatRecording) -> 'SpikeIntervalAssigner':
        """
        Create an instance of SpikeIntervalAssigner with the given episode names.
        """
        return SpikeIntervalAssigner(self._episode_names, recording)


class SpikeIntervalAssigner:
    """
    This class assigns the interval's name in which the spike occurs. Different
    types of episodes can't overlap.
    """

    def __init__(
        self,
        episode_names: list[str],
        recording: MatRecording
    ):
        self._episode_names = episode_names
        self._recording = recording

    def __call__(
        self,
        spikes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Mark the episode memberships in the features DataFrame.
        Args:
            spikes (pd.DataFrame): The DataFrame containing spike features.
            recording (MatRecording): The MatRecording object containing episode data.
        Returns:
        """
        timestamps = spikes['timestamp'].to_numpy()
        for episode_name in self._episode_names:
            spikes[episode_name] = self._mark_episode(
                timestamps,
                episode_name
            )
        return spikes

    def _mark_episode(
        self,
        timestamps: np.ndarray,
        episode_name: str
    ) -> np.ndarray:
        """
        Returns a binary integer array indicating whether each timestamp
        belongs to the specified episode.
        
        Args:
            timestamps (np.ndarray): The array containing spike timestamps.
            episode_name (str): The name of the episode to mark.
            
        Returns:
            np.ndarray: a binary array indicating episode membership or NaN
            if the episode is not present.
        """
        if f'{episode_name}_st' not in self._recording.list_markers():
            return np.full(len(timestamps), np.nan)
        episodes = self._recording.get_episode(episode_name)
        return self._create_episode_mask(timestamps, episodes).astype(int)

    def _create_episode_mask(
        self,
        timestamps: np.ndarray,
        episodes: EpisodeSequence
    ) -> np.ndarray:
        """
        Returns a boolean mask indicating for each timestamp whether
        it falls within any episode in the episode sequence.

        Args:
            timestamps (np.ndarray): The array containing spike timestamps.
            episodes (EpisodeSequence): The episode sequence to check against.

        Returns:
            np.ndarray: A boolean mask indicating whether each timestamp is within any episode.
        """
        masks = []
        for episode in episodes:
            masks.append(
                (episode.interval.start <= timestamps) &
                (timestamps <= episode.interval.end)
            )
        return np.logical_or.reduce(masks)


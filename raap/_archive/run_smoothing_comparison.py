"""
This script helps to compare SG smoothing parameters.
"""

import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from analyses.preprocessing import prepare_data
from data.spikes import SpikeShapes


def run_smoothing_comparison(
        spikes: SpikeShapes,
        window_lengths: list[int],
        poly_orders: list[int]
        ):
    """
    Run the smoothing comparison analysis on the input file.
    """
    filtered_spikes = {
        f'{window_length} {poly_order}': spikes.apply_sg_filter(
            window_length=window_length,
            polyorder=poly_order
        )
        for window_length in window_lengths
        for poly_order in poly_orders
    }

    grid_layout = (10, 1)
    spike_per_figure = grid_layout[0] * grid_layout[1]
    fig, axs = plt.subplots(
        grid_layout[0],
        grid_layout[1] * 2,
        figsize=(5, 20)
    )
    for i, spike in enumerate(spikes):
        ax_on_fig = i % spike_per_figure
        row = ax_on_fig // grid_layout[1]
        col = ax_on_fig % grid_layout[1]
        
        axs[row, col].plot(
            spike.times, spike.values,
            color='black', linewidth=2
        )
        axs[row, col + 1].plot(
            spike.values[:-1], np.diff(spike.values),
            color='black', linewidth=2
        )
        
        for window_length in window_lengths:
            for poly_order in poly_orders:
                sg_filtered = filtered_spikes[f'{window_length} {poly_order}'][i]
                axs[row, col].plot(
                    sg_filtered.times, sg_filtered.values,
                    linewidth=1,
                    label=f'{window_length} {poly_order}'
                )
                axs[row, col + 1].plot(
                    sg_filtered.values[:-1], np.diff(sg_filtered.values),
                    linewidth=1,
                    label=f'{window_length} {poly_order}'
                )

        axs[row, col + 1].legend()

        if ax_on_fig == 99 or i == len(spikes) - 1:
            plt.show()
            break

    # plt.savefig(input_path.with_suffix('.png'))

def run_smoothing_single_spike(
        spikes: SpikeShapes,
        window_lengths: list[int],
        poly_orders: list[int],
        spike_index: int = None
        ):
    """
    Run the smoothing comparison analysis on a single spike.
    """
    smoothed_spikes = {
        f'{window_length} {poly_order}': spikes.apply_sg_filter(
            window_length=window_length,
            polyorder=poly_order
        )
        for window_length in window_lengths
        for poly_order in poly_orders
    }

    row_count = len(poly_orders)
    col_count = len(window_lengths)
    fig, axs = plt.subplots(
        row_count,
        col_count,
        figsize=(row_count * 5, col_count * 5)
    )

    if not spike_index:
        spike_index = np.random.randint(0, len(spikes))

    spike = spikes[spike_index]

    for i, poly_order in enumerate(poly_orders):
        for j, window_length in enumerate(window_lengths):
            smooth_config_str = f'{window_length} {poly_order}'
            sg_filtered = smoothed_spikes[smooth_config_str][spike_index]

            axs[i, j].plot(
                spike.times, spike.values,
                color='#999999', linewidth=2
            )
            axs[i, j].plot(
                sg_filtered.times, sg_filtered.values,
                linewidth=1,
                color='#0000ff',
                label=f'{window_length} {poly_order}'
            )
            axs[i, j].set_title(smooth_config_str)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file',
        type=str,
        help='Path of the input file.'
    )
    parser.add_argument(
        'spike_config',
        type=str,
        help='Path of the spike extractor config file.'
    )

    args = parser.parse_args()

    with open(args.spike_config, 'r', encoding='utf-8') as f:
        spike_config = yaml.safe_load(f)

    input_path = Path(args.input_file)
    config = {
        'data': {
            'extractor': spike_config['extractor'],
            'preprocess': {
                'sg_filter': None,
                'normalize': None
            }
        }
    }

    spikes = prepare_data(input_path, config['data'])

    # run_smoothing_comparison(
    #     spikes,
    #     [5, 7, 9],
    #     [2, 3]
    # )

    run_smoothing_single_spike(
        spikes,
        [7,9, 11, 15],
        [1, 2, 3, 4]
    )

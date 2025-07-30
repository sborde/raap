from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.spatial.qhull import ConvexHull

from data.spikes import load_spike_shapes, SpikeShapes
from analyses.preprocessing import prepare_data


def calculate_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.
    """
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )

def calculate_concavity_index(polygon: np.ndarray, convex_hull: np.ndarray) -> float:
    """
    Calculate the concavity index of a polygon.
    """
    convex_area = calculate_area(convex_hull[:, 0], convex_hull[:, 1])
    polygon_area = calculate_area(polygon[:, 0], polygon[:, 1])
    return (convex_area - polygon_area) / convex_area if convex_area != 0 else 0

def calculate_individual_concavity_index(shapes: SpikeShapes):
    """
    Calculate the concavity index of a spike shape.
    """
    peak_i = shapes.peak
    concavity_indices = []
    for i, shape in enumerate(shapes):
        print(f'Processing {i + 1}/{len(shapes)}', end='\r')

        v = shape.values
        dv = np.diff(v)
        
        max_dv_i = np.argmax(dv)
        threshold_dv = 0.1 * np.max(dv)
        
        if not np.any(dv[:max_dv_i] < threshold_dv):
            concavity_index = np.nan
        else:
            start_i = np.where(dv[:max_dv_i] < threshold_dv)[0][-1] + 1
        
            if start_i is None or start_i > peak_i:
                concavity_index = np.nan
            else:
                phase_points = np.stack([
                    v[start_i:peak_i],
                    np.diff(v[start_i:peak_i+1])
                ], axis=1)

                convex_hull = ConvexHull(phase_points)
                hull_indices = convex_hull.vertices

                convex_hull = phase_points[hull_indices, :]

                concavity_index = calculate_concavity_index(phase_points, convex_hull)

        concavity_indices.append(concavity_index * 100)  # Convert to percentage

    return np.array(concavity_indices)

def project_to_umap(input_path: Path, config: dict, output_path: Path = None):
    """
    Project the spikes to the UMAP space.
    """
    spikes = prepare_data(input_path, config['data'])
    concavity_indices = calculate_individual_concavity_index(spikes)
    
    if config['data']['preprocess']['shape_part'] == 'half':
        spikes = spikes.get_half_spikes()

    umap_model = umap.UMAP(**config['umap'])
    umap_embeddings = umap_model.fit_transform(spikes.values)

    embeddings = pd.DataFrame(umap_embeddings, columns=['UMAP 1', 'UMAP 2'])
    embeddings['concavity_index'] = concavity_indices

    if output_path is not None:
        output_path = (
            output_path /
            f'n_{config["umap"]["n_neighbors"]}_'
            f'd_{config["umap"]["min_dist"]}_'
            f'm_{config["umap"]["metric"]}'
        )
        output_path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x='UMAP 1',
            y='UMAP 2',
            data=embeddings,
            hue='concavity_index',
            alpha=0.5
        )
        plt.title(f'{input_path.stem}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(False)
        plt.savefig(output_path / f'{input_path.stem}.png')
        plt.close()

    return umap_embeddings

def calculate_mean_deviation_index(input_path: Path, config: dict):
    """
    Perform the analysis.
    """
    spikes = load_spike_shapes(input_path, config['extractor'])

    spikes = spikes.apply_sg_filter()

    normalized_spikes = spikes.start_to_peak_normalized()
    half_spikes = normalized_spikes.get_half_spikes()

    median_half_spike_values = np.median(half_spikes.values, axis=0)
    q1_half_spike_values = np.percentile(half_spikes.values, 25, axis=0)
    q3_half_spike_values = np.percentile(half_spikes.values, 75, axis=0)
    
    upper_bound = median_half_spike_values + 1.5 * (q3_half_spike_values - q1_half_spike_values)

    output_path = Path(config['output_dir']) / input_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    grid_size = (3, 6)
    axis_per_plot = grid_size[0] * grid_size[1]

    fig, ax = plt.subplots(2 * grid_size[0], grid_size[1], figsize=(20, 20))
    for i, (spike, timestamp) in enumerate(zip(half_spikes, spikes.timestamps)):

        print(f'Processing {i} of {len(half_spikes)}', end='\r')

        fig_id = i // axis_per_plot
        axis_id = i % axis_per_plot

        row = axis_id // grid_size[1]
        col = axis_id % grid_size[1]

        ax[2 * row, col].plot(half_spikes.times, median_half_spike_values, c='black')
        ax[2 * row, col].fill_between(
            half_spikes.times,
            q1_half_spike_values,
            q3_half_spike_values,
            color='black', alpha=0.5
        )
        ax[2 * row, col].plot(half_spikes.times, upper_bound, c='black', linestyle='--')
        ax[2 * row, col].plot(spike.times, spike.values, c='red')
        ax[2 * row, col].set_title(f'{timestamp:.2f} s')
        ax[2 * row, col].set_xticks([])
        ax[2 * row, col].set_yticks([])
        ax[2 * row, col].grid(False)

        # ax[2 * row + 1, col].plot(spike.values[:-1], np.diff(spike.values), c='red')
        # ax[2 * row + 1, col].set_xticks([])
        # ax[2 * row + 1, col].set_yticks([])
        # ax[2 * row + 1, col].grid(False)
        ax[2 * row + 1, col].plot(spike.times[:-1], np.diff(spike.values), c='green')
        ax[2 * row + 1, col].set_xticks([])
        ax[2 * row + 1, col].set_yticks([])
        ax[2 * row + 1, col].grid(False)

        if axis_id == axis_per_plot - 1 or i == len(half_spikes) - 1:
            plt.tight_layout()
            plt.savefig(output_path / f'{fig_id:03d}.png')
            plt.close()
            fig, ax = plt.subplots(2 * grid_size[0], grid_size[1], figsize=(20, 20))




from pathlib import Path

from raap.data.spikeshapes import load_spike_shapes, SpikeShape


def load_cell_spikes(
    path: str | Path,
    config: dict
):
    spikes = load_spike_shapes(path, **config)


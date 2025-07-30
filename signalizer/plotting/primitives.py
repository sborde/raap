import abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt


class VectorPairPlotter(abc.ABC):

    @abc.abstractmethod
    def plot(self, axis: plt.Axes, x: np.ndarray, y: np.ndarray): pass

    @staticmethod
    def from_config(config: dict) -> 'VectorPairPlotter':
        if config['type'] == 'scatter':
            # return ScatterPlotter.from_config(config)
            raise NotImplementedError("ScatterPlotter is not implemented yet.")
        elif config['type'] == 'line':
            return LinePlotter.from_config(config)
        else:
            raise ValueError(f"Unsupported plot type: {config['type']}")


class LinePlotter(VectorPairPlotter):
    def __init__(self, color=None, width=1, style='-'):
        self._color = color if color else 'k'
        self._width = width
        self._style = style

    def with_color(self, color) -> 'LinePlotter':
        return LinePlotter(color=color, width=self._width)

    def with_thickness(self, thickness) -> 'LinePlotter':
        return LinePlotter(color=self._color, width=thickness)

    def with_style(self, style) -> 'LinePlotter':
        return LinePlotter(color=self._color, width=self._width, style=style)

    def plot(self, axis: plt.Axes, x: np.ndarray, y: np.ndarray):
        axis.plot(x, y, self._style, color=self._color, linewidth=self._width)

    @classmethod
    def from_config(cls, config: dict) -> 'LinePlotter':
        color = config.get('color', None)
        width = config.get('width', 1)
        style = config.get('style', '-')
        return cls(color=color, width=width, style=style)

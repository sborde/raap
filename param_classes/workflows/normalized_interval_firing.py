"""This module contains parameter classes for normalized interval firing workflows."""

from pydantic import BaseModel

from param_classes.workflows.input_output import GroupInfoParams, InputDataParams, OutputParams


class NormalizedTimeHistogramParams(BaseModel):
    """Parameter class for histogram calculation."""
    bin_range: tuple[float, float] = (0.0, 0.0)
    bin_size: float = 0.0
    def __init__(
        self,
        bin_range: tuple[float, float] = (0.0, 0.0),
        bin_size: float = 0.0,
        value_unit: str = 'freq',
    ):
        super().__init__(
            bin_range=bin_range,
            bin_size=bin_size
        )
        self._value_unit = value_unit

    @property
    def value_unit(self) -> str:
        """Get the value unit."""
        return self._value_unit
    @value_unit.setter
    def value_unit(self, value: str) -> None:
        """Set the value unit."""
        if value not in ['count', 'freq', 'prob']:
            raise ValueError("value_unit must be one of 'count', 'freq', or 'prob'")
        self._value_unit = value


class NormalizedIntervaFiringAnalysisParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    values: str = 'freq'
    bin_size: float = 0.0
    bin_range: tuple[float, float] = (0.0, 0.0)
    smooth_size: float = 0.0
    episode_name: str = ''
    spike_types: list[str] = []
    interval_condition: list[str] = []
    show_raster: bool = True
    raster_size: float = 0.5
    raster_band_color: str = '#ababab'
    ylim: tuple[float, float] = (0.0, 0.0)

class NormalizedIntervalFiringWorkflowParams(BaseModel):
    """Parameter class for the normalized inter RA-AP firing workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis: NormalizedIntervaFiringAnalysisParams = NormalizedIntervaFiringAnalysisParams()
    grouping: GroupInfoParams = GroupInfoParams()

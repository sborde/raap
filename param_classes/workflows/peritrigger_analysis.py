"""Parameter classes for the peri-trigger analysis workflow."""

from pydantic import BaseModel

from param_classes.workflows.input_output import InputDataParams, OutputParams

class HistogramParams(BaseModel):
    """Parameter class for histogram calculation."""
    bin_range: tuple[float, float] = (0.0, 0.0)
    bin_size: float = 0.0
    def __init__(
        self,
        bin_range: tuple[float, float] = (0.0, 0.0),
        bin_size: float = 0.0,
        data_unit: str = 'ms',
        value_unit: str = 'freq',
    ):
        super().__init__(
            bin_range=bin_range,
            bin_size=bin_size
        )
        self._data_unit = data_unit
        self._value_unit = value_unit

    @property
    def data_unit(self) -> str:
        """Get the data unit."""
        return self._data_unit
    @data_unit.setter
    def data_unit(self, value: str) -> None:
        """Set the data unit."""
        if value not in ['s', 'ms']:
            raise ValueError("data_unit must be 's' or 'ms'")
        self._data_unit = value

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


class PeriTriggerAnalysisParams(BaseModel):
    """Parameter class for the peri-trigger analysis workflow."""
    trigger: str = ''
    spike_types: list[str] = []
    histogram: HistogramParams = HistogramParams()
    smooth_size: float = 0.0
    ylim: tuple[float, float] = (0.0, 0.0)


class PeriTriggerLatencyComparisonParams(BaseModel):
    """Parameter class for groupwise comparison settings."""
    trigger_condition: list[str] = []
    detection_start: float = 0.0
    boxplot_ylim: tuple[float, float] = (0.0, 0.0)


class PeriTriggerAnalysisWorkflowParams(BaseModel):
    """Parameter class for the episode analysis workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis: PeriTriggerAnalysisParams = PeriTriggerAnalysisParams()
    grouping: PeriTriggerLatencyComparisonParams = PeriTriggerLatencyComparisonParams()

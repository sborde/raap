"""Parameter classes for the normalilzed time interval analysis workflow."""


from pydantic import BaseModel

from param_classes.workflows.input_output import InputDataParams, OutputParams


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


class InterMarkerIntervalAnalysis(BaseModel):
    """Parameter class for the inter-marker interval analysis workflow."""
    histogram: NormalizedTimeHistogramParams = NormalizedTimeHistogramParams()
    smooth_size: float = 0.0
    ylim: tuple[float, float] = (0.0, 0.0)

class NormalizedInterRaapFiringWorkflowParams(BaseModel):
    """Parameter class for the normalized inter RA-AP firing workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis: InterMarkerIntervalAnalysis = InterMarkerIntervalAnalysis()


class InterRaapSpikeCountAnalysis(BaseModel):
    """Parameter class for the inter RA-AP spike count analysis."""
    episodes: list[str] = []


class InterRaapSpikeCountFiringWorkflowParams(BaseModel):
    """Parameter class for the inter RA-AP spike count firing workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis : InterRaapSpikeCountAnalysis = InterRaapSpikeCountAnalysis()

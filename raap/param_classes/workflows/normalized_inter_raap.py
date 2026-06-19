"""Parameter classes for the normalilzed time interval analysis workflow."""


from pydantic import BaseModel

from param_classes.workflows.input_output import InputDataParams, OutputParams
from .normalized_interval_firing import NormalizedTimeHistogramParams


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

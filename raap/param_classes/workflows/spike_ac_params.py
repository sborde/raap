"""This module contains parameter classes for normalized interval firing workflows."""

from typing import Optional
from pydantic import BaseModel

from param_classes.workflows.input_output import InputDataParams, OutputParams



class SpikeACAnalysisParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    values: str = 'freq'
    bin_size: float = 0.0
    bin_range: tuple[float, float] = (0.0, 0.0)
    smooth_size: float = 0.0
    episodes: list[str] = []
    spike_types: list[str] = []
    show_raster: bool = False
    raster_size: float = 0.5
    to_svg: bool = False
    ylim: Optional[tuple[float, float]] = None

class SpikeACAnalysisParamsWorkflowParams(BaseModel):
    """Parameter class for the spike AC analysis workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis: SpikeACAnalysisParams = SpikeACAnalysisParams()

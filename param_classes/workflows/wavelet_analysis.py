"""This module contains the parameter classes for episode analysis."""

from pydantic import BaseModel

from param_classes.workflows.input_output import GroupInfoParams, InputDataParams, OutputParams

class WaveletParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    low: float = 0.0
    high: float = 0.0
    step: float = 1.0
    n_cycles: int = 7
    length: float = 1.0
    addtaper: bool = False
    taperlength: float = 0.1

class SignalParameters(BaseModel):
    """Parameter class for signal parameters."""
    name: str = 'wideband'
    fs: int = 1000

class WaveletAnalysisParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    wavelet: WaveletParams = WaveletParams()
    signal: SignalParameters = SignalParameters()

class SpikeTriggeredWaveletAnalysisParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    spike_types: list[str] = []
    cut_window: tuple[float, float] = (-0.5, 0.5)
    signal: SignalParameters = SignalParameters()
    wavelet: WaveletParams = WaveletParams()
    clim: tuple[float, float] = (0.0, 0.0)
    xlim: tuple[float, float] = (0.0, 0.0)
    ylim: tuple[float, float] = (0.0, 0.0)
    to_png: bool = False
    to_svg: bool = False
    dpi: int = 300

class WindowedWaveletAnalysisParams(BaseModel):
    """Parameter class for the normalized interval firing analysis."""
    spike_types: list[str] = []
    window_size: float = 1.0
    step_size: float = 0.1
    signal: SignalParameters = SignalParameters()
    wavelet: WaveletParams = WaveletParams()
    clim: tuple[float, float] = (0.0, 0.0)
    xlim: tuple[float, float] = (0.0, 0.0)
    ylim: tuple[float, float] = (0.0, 0.0)
    to_png: bool = False
    to_svg: bool = False
    dpi: int = 300

class SpikeTriggeredWaveletWorkflowParams(BaseModel):
    """Parameter class for the normalized inter RA-AP firing workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    analysis: SpikeTriggeredWaveletAnalysisParams = SpikeTriggeredWaveletAnalysisParams()
    grouping: GroupInfoParams = GroupInfoParams()

class WorkflowParams[T](BaseModel):
    """Parameter class for the normalized inter RA-AP firing workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    palette: dict = {}
    output: OutputParams = OutputParams()
    grouping: GroupInfoParams = GroupInfoParams()
    analysis: T

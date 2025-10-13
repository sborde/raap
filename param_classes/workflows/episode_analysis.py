"""This module contains the parameter classes for episode analysis."""

from typing import Any, Callable, Literal
from pydantic import BaseModel


class FigureSaveParams(BaseModel):
    """Parameter class for figure settings."""
    dpi: int = 100
    as_png: bool = True
    as_svg: bool = False

    @property
    def save(self) -> bool:
        """Determine if any figure saving is enabled."""
        return self.as_png or self.as_svg


class CellwiseResultsPathParams(BaseModel):
    """Parameter class for results path settings."""
    path: str = ''
    extension: str = ''
    suffix: str = ''


class RawDataPathParams(BaseModel):
    """Parameter class for data path settings."""
    path: str = ''
    extension: str = 'mat'
    suffix: str = ''


class SpikeFeaturesPathParam(BaseModel):
    """Parameter class for the path of spike features"""
    path: str = ''
    extension: str = 'csv'
    suffix: str = ''

    def model_dump(self) -> dict:
        return {
            'results': super().model_dump()
        }

class InputDataParams(BaseModel):
    """Parameter class for input data settings."""
    raw_data: RawDataPathParams = RawDataPathParams()
    spike_features: SpikeFeaturesPathParam = SpikeFeaturesPathParam()

    def model_dump(self) -> dict:
        return {
            'raw_data': self.raw_data.model_dump()
        } | self.spike_features.model_dump()


class GroupingInputParams(BaseModel):
    """Parameter class for grouping input settings."""
    results: CellwiseResultsPathParams = CellwiseResultsPathParams()


class OutputParams(BaseModel):
    """Parameter class for output settings."""
    path: str = ''
    figures: FigureSaveParams = FigureSaveParams()

class EpisodeAnalysisParams(BaseModel):
    """Parameter class for episode analysis."""
    episodes: list[str] = []
    spike_types: list[str] = []
    suffix: str = 'episodes'
    verbose: bool = False

    def __init__(
        self,
        episodes: list[str] = [],
        spike_types: list[str] = [],
        suffix: str = 'episodes',
        verbose: bool = False,
        interval_type: str = ''
    ):
        super().__init__(
            episodes=episodes,
            spike_types=spike_types,
            suffix=suffix,
            verbose=verbose
        )
        self._interval_type = interval_type
    @property
    def interval_type(self) -> str:
        """Get the interval type."""
        if self._interval_type == '':
            raise ValueError("interval_type has not been set")
        return self._interval_type
    @interval_type.setter
    def interval_type(self, value: str) -> None:
        """Set the interval type."""
        if value not in ['episode', 'cycle', 'window']:
            raise ValueError("interval_type must be 'episode', 'cycle', or 'window'")
        self._interval_type = value

class GroupingParams(BaseModel):
    """Parameter class for grouping settings."""
    aggregation: str =  'mean'
    condition: list[str] = []
    group_order: list[str] = []

class EpisodeAnalysisWorkflowParams(BaseModel):
    """Parameter class for the episode analysis workflow."""
    data: InputDataParams = InputDataParams()
    input: dict = {}
    output: OutputParams = OutputParams()
    analysis: EpisodeAnalysisParams = EpisodeAnalysisParams()


class GroupwiseComparisonParams(BaseModel):
    """Parameter class for groupwise comparison settings."""
    palette: dict = {}
    data: GroupingInputParams = GroupingInputParams()
    input: dict = {}
    output: OutputParams = OutputParams()
    analysis: EpisodeAnalysisParams = EpisodeAnalysisParams()
    grouping: GroupingParams = GroupingParams()
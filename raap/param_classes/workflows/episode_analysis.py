"""This module contains the parameter classes for episode analysis."""

from pydantic import BaseModel

from param_classes.workflows.input_output import GroupingInputParams, InputDataParams, OutputParams


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

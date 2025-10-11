"""This module contains the parameter classes for episode analysis."""

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

class GroupingParams(BaseModel):
    """Parameter class for grouping settings."""
    aggregation: str =  'mean'
    group_order: list[str] = []

class EpisodeAnalysisWorkflowParams(BaseModel):
    """Parameter class for the episode analysis workflow."""
    palette: dict = {}
    data: dict = {}
    input: dict = {}
    output: OutputParams = OutputParams()
    analysis: EpisodeAnalysisParams = EpisodeAnalysisParams()
    grouping: GroupingParams = GroupingParams()
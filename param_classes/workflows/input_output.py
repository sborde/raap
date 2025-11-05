"""A module for input and output parameter classes."""

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


class OutputParams(BaseModel):
    """Parameter class for output settings."""
    path: str = ''
    figures: FigureSaveParams = FigureSaveParams()


class InputDataParams(BaseModel):
    """Parameter class for input data settings."""
    raw_data: RawDataPathParams = RawDataPathParams()
    spike_features: SpikeFeaturesPathParam = SpikeFeaturesPathParam()

    def model_dump(self) -> dict:
        return {
            'raw_data': self.raw_data.model_dump()
        } | self.spike_features.model_dump()


class GroupInfoParams(BaseModel):
    """Parameter class for grouping information settings."""
    path: str = ''
    column_name: str = ''
    groups: list[str] = []

class GroupingInputParams(BaseModel):
    """Parameter class for grouping input settings."""
    results: CellwiseResultsPathParams = CellwiseResultsPathParams()

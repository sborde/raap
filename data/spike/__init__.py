
from typing import Optional
from .transformers import (
    SpikeTransformer,
    DummySpikeTransformer,
    SpikeStartAligner
)
from .processors import SavGolFilter
from .detectors import SpikeStartFinder

def create_transformer(
    name: str,
    params: Optional[dict]  = None
) -> SpikeTransformer:
    if name == 'align':
        if params is None:
            raise ValueError(
                'Parameters must be provided ' +
                'for \'align\' transformer.'
            )
        start_finder = create_start_finder(params)
        return SpikeStartAligner(start_finder)
    else:
        return DummySpikeTransformer()

def create_start_finder(params: dict) -> SpikeStartFinder:
    """
    Factory function to create a SpikeStartFinder instance with given parameters.
    
    Args:
        params (dict): Parameters for the start finder, including 'start_threshold'
                       and optional 'smooth_params'.
    
    Returns:
        SpikeStartFinder: An instance of SpikeStartFinder configured with the provided parameters.
    """
    smoother = None
    if params.get('smooth_params', None):
        smoother = SavGolFilter(**params['smooth_params'])
    return SpikeStartFinder(
        params['start_threshold'],
        smoother
    )
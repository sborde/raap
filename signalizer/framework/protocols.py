import abc
from typing import Protocol

from signalizer.framework.pipeline.protocols import PipelineBlock


class SavableOperator(Protocol):

    @property
    @abc.abstractmethod
    def params(self) -> dict: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> 'SavableOperator': ...


class Processor[T](SavableOperator, Protocol):

    @abc.abstractmethod
    def process(self, data: T) -> T: ...



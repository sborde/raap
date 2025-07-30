import abc
from typing import Optional

from signalizer.model.wrappers import EntityData


class ProgressListener(abc.ABC):

    @abc.abstractmethod
    def start(self, iter_count: int): pass

    @abc.abstractmethod
    def update(self, item = None): pass

    @abc.abstractmethod
    def finish(self): pass


class ProgressNotifier(abc.ABC):

    @abc.abstractmethod
    def add_listener(self, listener: ProgressListener): pass


class ConsoleListener(ProgressListener):
    """A simple console listener for progress updates."""
    def __init__(self, label: Optional[str] = None):
        self._label = label or 'Progress'
        self._total = 0
        self._current = 0

    def start(self, total: int) -> None:
        self._current = 0
        self._total = total

    def update(self, item = None) -> None:
        self._current += 1
        percent = int((self._current / self._total) * 100)
        label = self._label
        if item:
            if isinstance(item, EntityData):
                label += f' ({item.entity.name})'
            else:
                label += f' ({item})'
        print(f'[{"|" * percent}{"." * (100 - percent)}] {percent}% ({self._current}/{self._total}) - {label}', end='\r')

    def finish(self) -> None:
        print(f'{self._label} finished.')


class ProgressIterator[T]:

    def __init__(self, data, listeners: list[ProgressListener] = None):
        self._data = data
        self._count = len(data)
        self._index = 0
        self._listeners = listeners or []

    def __iter__(self):
        for listener in self._listeners:
            listener.start(self._count)
        return self

    def __next__(self):
        if self._index >= self._count:
            for listener in self._listeners:
                listener.finish()
            raise StopIteration

        item = self._data[self._index]
        self._index += 1

        for listener in self._listeners:
            listener.update(item)

        return item

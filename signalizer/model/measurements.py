from enum import Enum


class TimeUnit(Enum):
    SECONDS = ("s", 1.0)
    MILLISECONDS = ("ms", 0.001)

    def __init__(self, abbreviation, conversion_factor):
        self.abbreviation = abbreviation
        self.conversion_factor = conversion_factor

    def __str__(self) -> str:
        return self.abbreviation

    def convert(self, value, new_unit):
        if self == new_unit:
            return value
        else:
            seconds = self.to_seconds(value)
            return new_unit.from_seconds(seconds)

    def to_seconds(self, value):
        return value * self.conversion_factor

    def from_seconds(self, seconds):
        return seconds / self.conversion_factor

    @classmethod
    def from_abbreviation(cls, abbreviation):
        for unit in cls:
            if unit.abbreviation == abbreviation:
                return unit
        raise ValueError(f"Unknown time unit: {abbreviation}")


class VoltageUnit(Enum):
    VOLTS = ("V", 1.0)
    MILLIVOLTS = ("mV", 0.001)

    def __init__(self, abbreviation, conversion_factor):
        self.abbreviation = abbreviation
        self.conversion_factor = conversion_factor

    def __str__(self) -> str: return self.abbreviation

    def to_volts(self, value):
        return value * self.conversion_factor

    def from_volts(self, volts):
        return volts / self.conversion_factor


class FrequencyUnit(Enum):
    HERTZ = ("Hz", 1.0)
    KILOHERTZ = ("kHz", 1000.0)

    def __init__(self, abbreviation, conversion_factor):
        self.abbreviation = abbreviation
        self.conversion_factor = conversion_factor

    def to_hertz(self, value):
        return value * self.conversion_factor

    def from_hertz(self, hertz):
        return hertz / self.conversion_factor
import abc
import dataclasses
import enum
from typing import ClassVar


class ColorSpace(enum.Enum):
    CIEXYZ = "CIEXYZ"
    SRGB = "SRGB"
    CIELAB = "CIELAB"
    CIELUV = "CIELUV"
    HSL = "HSL"


@dataclasses.dataclass(frozen=True)
class Color(abc.ABC):
    values: list[float]
    _: dataclasses.KW_ONLY
    _space: ColorSpace | str = dataclasses.field(init=False)
    _value_names: ClassVar[list[str]] = dataclasses.field(init=False)

    def __post_init__(self):
        if not isinstance(self._space, ColorSpace):
            object.__setattr__(self, "_space", ColorSpace(self.space))
        if not len(self.values) == len(self._value_names):
            raise ValueError(
                "Length of values must match length of value names. "
                f"Got {len(self.values)} values and expected {len(self._value_names)}."
            )
        
    def __str__(self):
        repr_values = ", ".join(f"{name}={value}" for name, value in zip(self._value_names, self.values))
        return f"{self.__class__.__name__}({repr_values})"
    

@dataclasses.dataclass(frozen=True)
class SRGBColor(Color):
    _space: ClassVar[ColorSpace] = ColorSpace.SRGB
    _value_names: ClassVar[list[str]] = ["R", "G", "B"]

    def __post_init__(self):
        super().__post_init__()
        if not all(0 <= value <= 1 for value in self.values):
            raise ValueError("Values must be between 0 and 1.")


@dataclasses.dataclass(frozen=True)
class HSLColor(Color):
    _space: ClassVar[ColorSpace] = ColorSpace.HSL
    _value_names: ClassVar[list[str]] = ["H", "S", "L"]

    def __post_init__(self):
        super().__post_init__()
        if not all(0 <= value <= 1 for value in self.values[1:]):
            raise ValueError("Values must be between 0 and 1.")
        if not 0 <= self.values[0] <= 360:
            raise ValueError("Hue must be between 0 and 360.")


@dataclasses.dataclass(frozen=True)
class CIEXYZColor(Color):
    _space: ClassVar[ColorSpace] = ColorSpace.CIEXYZ
    _value_names: ClassVar[list[str]] = ["X", "Y", "Z"]


@dataclasses.dataclass(frozen=True)
class CIELABColor(Color):
    _space: ClassVar[ColorSpace] = ColorSpace.CIELAB
    _value_names: ClassVar[list[str]] = ["L", "a", "b"]


@dataclasses.dataclass(frozen=True)
class CIELUVColor(Color):
    _space: ClassVar[ColorSpace] = ColorSpace.CIELUV
    _value_names: ClassVar[list[str]] = ["L", "u", "v"]

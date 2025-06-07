from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Target:
    confidence: float
    center: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    box: tuple[int, int, int, int] | None = None
    is_lost: bool = True


@dataclass(frozen=True)
class TargetIBVS(Target):
    masks_xy: list[list[tuple[int, int]]] | np.ndarray | None = None
    bbox_oriented: list[tuple[int, ...]] | None = None

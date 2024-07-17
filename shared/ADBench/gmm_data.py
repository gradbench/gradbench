# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/GMMData.py

"""
Changes Made:
- Reformatted all "np.ndarray = field(default = np.empty(0, dtype = np.float64))" to "np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))"
"""

from dataclasses import dataclass, field

import numpy as np
from defs import Wishart


@dataclass
class GMMInput:
    alphas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    means: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    icf: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    wishart: Wishart = field(default_factory=lambda: Wishart())


@dataclass
class GMMOutput:
    objective: np.float64 = 0.0
    gradient: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass
class GMMParameters:
    replicate_point: bool = False

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/HandData.py

"""
Changes Made:
- Changed default= to default_factory in data classes.
- Added dataclass_json decorators.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass
class HandModel:
    bone_count: int = field(default=0)

    bone_names: Tuple[str, ...] = field(default=tuple())

    # asssuming that parent is earlier in the order of bones
    parents: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=config(
            encoder=lambda x: x.tolist(),
            decoder=lambda x: np.asarray(x, dtype=np.int32),
        ),
    )

    base_relatives: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

    inverse_base_absolutes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

    base_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

    weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

    # two dimensional array with the second dimension equals to 3
    triangles: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=config(
            encoder=lambda x: x.tolist(),
            decoder=lambda x: np.asarray(x, dtype=np.int32),
        ),
    )

    is_mirrored: bool = field(default=False)


@dataclass_json
@dataclass
class HandData:
    model: HandModel = field(default_factory=lambda: HandModel())

    correspondences: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=config(
            encoder=lambda x: x.tolist(),
            decoder=lambda x: np.asarray(x, dtype=np.int32),
        ),
    )

    points: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )


@dataclass_json
@dataclass
class HandInput:
    theta: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )
    data: HandData = field(default_factory=lambda: HandData())
    us: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )


@dataclass_json
@dataclass
class HandOutput:
    objective: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )
    jacobian: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

    def save_output_to_file(self, output_prefix, input_basename, module_basename):
        save_vector_to_file(
            objective_file_name(output_prefix, input_basename, module_basename),
            self.objective,
        )

        save_jacobian_to_file(
            jacobian_file_name(output_prefix, input_basename, module_basename),
            self.jacobian,
        )


@dataclass_json
@dataclass
class HandParameters:
    is_complicated: bool = field(default=False)

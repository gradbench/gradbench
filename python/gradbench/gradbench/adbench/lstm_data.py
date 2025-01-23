# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/LSTMData.py

"""
Changes Made:
- Changed default= to default_factory in data classes.
- Added dataclass_json decorators.
"""


from dataclasses import dataclass, field

import numpy as np
from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass
class LSTMInput:
    main_params: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )
    extra_params: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )
    state: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )
    sequence: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )


@dataclass_json
@dataclass
class LSTMOutput:
    objective: np.float64 = 0.0
    gradient: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        metadata=config(encoder=lambda x: x.tolist(), decoder=np.asarray),
    )

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/utils.cpp

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <functional>
#include <limits>

#include "defs.h"

// Time a function
double timer(int nruns, double limit, std::function<void()> func);

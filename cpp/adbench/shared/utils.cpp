// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/utils.cpp
//
// Changes made: removed most code.

#include "utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <functional>
#include <limits>
#include <chrono>
#include <cstring>

#include "light_matrix.h"

#include "defs.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::getline;
using std::memcpy;
using namespace std::chrono;

// Time a function
double timer(int nruns, double limit, std::function<void()> func) {
    if (limit < 0) limit = std::numeric_limits<double>::max();

    double total = 0;
    int i = 0;

    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (; i < nruns && total < limit; ++i) {
        func();
        high_resolution_clock::time_point end = high_resolution_clock::now();
        total = duration_cast<duration<double>>(end - start).count();
    }

    if (i < nruns) std::cout << "Hit time limit after " << i << " loops" << endl;

    if (i > 0)
        return total / i;
    else
        return 0;
}

#pragma GCC diagnostic pop

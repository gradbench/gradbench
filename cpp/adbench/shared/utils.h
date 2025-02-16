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

#include "light_matrix.h"

#include "defs.h"

class HandModelLightMatrix
{
public:
    std::vector<std::string> bone_names;
    std::vector<int> parents; // assumimng that parent is earlier in the order of bones
    std::vector<LightMatrix<double>> base_relatives;
    std::vector<LightMatrix<double>> inverse_base_absolutes;
    LightMatrix<double> base_positions;
    LightMatrix<double> weights;
    std::vector<Triangle> triangles;
    bool is_mirrored;
};

class HandDataLightMatrix
{
public:
    HandModelLightMatrix model;
    std::vector<int> correspondences;
    LightMatrix<double> points;
};

#ifdef DO_EIGEN
void read_hand_model(const std::string& path, HandModelEigen* pmodel);

void read_hand_instance(const std::string& model_dir, const std::string& fn_in,
    std::vector<double>* theta, HandDataEigen* data, std::vector<double>* us = nullptr);
#endif

void read_hand_model(const std::string& path, HandModelLightMatrix* pmodel);

void read_hand_instance(const std::string& model_dir, const std::string& fn_in,
    std::vector<double>* theta, HandDataLightMatrix* data, std::vector<double>* us = nullptr);

// Time a function
double timer(int nruns, double limit, std::function<void()> func);

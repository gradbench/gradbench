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

#ifdef DO_EIGEN
#include "hand_eigen_model.h"
#endif


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


template<class T>
class precise_ofstream : public std::ofstream
{
public:
    precise_ofstream(std::string fn):std::ofstream(fn) {
        auto max_digits = std::numeric_limits<T>::max_digits10;
        this->precision(max_digits);
        *this << std::scientific;
    }
};

template<class T>
class write_J_stream : public precise_ofstream<T>
{
public:
    write_J_stream(std::string fn, size_t rows, size_t cols)
        :precise_ofstream<T>(fn)
    {
        std::cout << "Writing to " << fn << std::endl;
        if (!this->good()) {
            std::cerr << "FAILED\n";
            throw "oik";
        }
        *this << rows << " " << cols << std::endl;
    }
};


void write_J(const std::string& fn, int Jrows, int Jcols, double** J);

void write_J(const std::string& fn, int Jrows, int Jcols, double* J);

void write_times(double tf, double tJ);

void write_times(const std::string& fn, double tf, double tJ, double* t_sparsity = nullptr);

#ifdef DO_EIGEN
void read_hand_model(const std::string& path, HandModelEigen* pmodel);

void read_hand_instance(const std::string& model_dir, const std::string& fn_in,
    std::vector<double>* theta, HandDataEigen* data, std::vector<double>* us = nullptr);
#endif

void read_hand_model(const std::string& path, HandModelLightMatrix* pmodel);

void read_hand_instance(const std::string& model_dir, const std::string& fn_in,
    std::vector<double>* theta, HandDataLightMatrix* data, std::vector<double>* us = nullptr);

void read_lstm_instance(const std::string& fn,
    int* l, int* c, int* b,
    std::vector<double>& main_params,
    std::vector<double>& extra_params,
    std::vector<double>& state,
    std::vector<double>& sequence);

// Time a function
double timer(int nruns, double limit, std::function<void()> func);

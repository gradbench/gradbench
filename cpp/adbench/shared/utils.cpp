// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/utils.cpp
//
// Changes made: silenced some warnings, removed unnecessary code.

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

void read_hand_model(const string& path, HandModelLightMatrix* pmodel)
{
    const char DELIMITER = ':';
    auto& model = *pmodel;
    std::ifstream bones_in(path + "bones.txt");
    string s;
    while (bones_in.good())
    {
        getline(bones_in, s, DELIMITER);
        if (s.empty())
            continue;
        model.bone_names.push_back(s);
        getline(bones_in, s, DELIMITER);
        model.parents.push_back(std::stoi(s));
        double tmp[16];
        for (int i = 0; i < 16; i++)
        {
            getline(bones_in, s, DELIMITER);
            tmp[i] = std::stod(s);
        }
        model.base_relatives.emplace_back(4, 4);
        model.base_relatives.back().set(tmp);
        model.base_relatives.back().transpose_in_place();
        for (int i = 0; i < 15; i++)
        {
            getline(bones_in, s, DELIMITER);
            tmp[i] = std::stod(s);
        }
        getline(bones_in, s, '\n');
        tmp[15] = std::stod(s);
        model.inverse_base_absolutes.emplace_back(4, 4);
        model.inverse_base_absolutes.back().set(tmp);
        model.inverse_base_absolutes.back().transpose_in_place();
    }
    bones_in.close();
    int n_bones = (int)model.bone_names.size();

    std::ifstream vert_in(path + "vertices.txt");
    int n_vertices = 0;
    while (vert_in.good())
    {
        getline(vert_in, s);
        if (!s.empty())
            n_vertices++;
    }
    vert_in.close();

    model.base_positions.resize(4, n_vertices);
    model.base_positions.set_row(3, 1.);
    model.weights.resize(n_bones, n_vertices);
    model.weights.fill(0.);
    vert_in = std::ifstream(path + "vertices.txt");
    for (int i_vert = 0; i_vert < n_vertices; i_vert++)
    {
        for (int j = 0; j < 3; j++)
        {
            getline(vert_in, s, DELIMITER);
            model.base_positions(j, i_vert) = std::stod(s);
        }
        for (int j = 0; j < 3 + 2; j++)
        {
            getline(vert_in, s, DELIMITER); // skip
        }
        getline(vert_in, s, DELIMITER);
        int n = std::stoi(s);
        for (int j = 0; j < n; j++)
        {
            getline(vert_in, s, DELIMITER);
            int i_bone = std::stoi(s);
            if (j == n - 1)
                getline(vert_in, s, '\n');
            else
                getline(vert_in, s, DELIMITER);
            model.weights(i_bone, i_vert) = std::stod(s);
        }
    }
    vert_in.close();

    std::ifstream triangles_in(path + "triangles.txt");
    string ss[3];
    while (triangles_in.good())
    {
        getline(triangles_in, ss[0], DELIMITER);
        if (ss[0].empty())
            continue;

        getline(triangles_in, ss[1], DELIMITER);
        getline(triangles_in, ss[2], '\n');
        Triangle curr;
        for (int i = 0; i < 3; i++)
            curr.verts[i] = std::stoi(ss[i]);
        model.triangles.push_back(curr);
    }
    triangles_in.close();

    model.is_mirrored = false;
}

void read_hand_instance(const string& model_dir, const string& fn_in,
    vector<double>* theta, HandDataLightMatrix* data, vector<double>* us)
{
    read_hand_model(model_dir, &data->model);
    std::ifstream in(fn_in);
    int n_pts, n_theta;
    in >> n_pts >> n_theta;
    data->correspondences.resize(n_pts);
    data->points.resize(3, n_pts);
    for (int i = 0; i < n_pts; i++)
    {
        in >> data->correspondences[i];
        for (int j = 0; j < 3; j++)
        {
            in >> data->points(j, i);
        }
    }
    if (us != nullptr)
    {
        us->resize(2 * n_pts);
        for (int i = 0; i < 2 * n_pts; i++)
        {
            in >> (*us)[i];
        }
    }
    theta->resize(n_theta);
    for (int i = 0; i < n_theta; i++)
    {
        in >> (*theta)[i];
    }
    in.close();
}

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

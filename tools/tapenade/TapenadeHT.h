// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeHand.h

#pragma once

#include <vector>

#include "adbench/shared/ITest.h"
#include "adbench/shared/HandData.h"
#include "adbench/shared/light_matrix.h"
#include "adbench/shared/defs.h"

#include "hand/hand_d.h"

// Data for hand objective converted from input
struct HandObjectiveData
{
    int bone_count;
    const char** bone_names;
    const int* parents;             // assumimng that parent is earlier in the order of bones
    Matrix* base_relatives;
    Matrix* inverse_base_absolutes;
    Matrix base_positions;
    Matrix weights;
    const Triangle* triangles;
    int is_mirrored;
    int corresp_count;
    const int* correspondences;
    Matrix points;
};

class TapenadeHand : public ITest<HandInput, HandOutput> {
    HandObjectiveData* objective_input = nullptr;
    HandInput input;
    HandOutput result;
    bool complicated = false;

    std::vector<double> theta_d;                // buffer for theta differentiation directions
    std::vector<double> us_d;                   // buffer for us differentiation directions
    Matrix_diff *base_relativesd ;              // buffer needed to hold the diff of objective_input->base_relatives
    Matrix_diff *inverse_base_absolutesd ;      // buffer needed to hold the diff of objective_input->inverse_base_absolutes
    Matrix_diff base_positionsd ;               // buffer needed to hold the diff of objective_input->base_positions

    std::vector<double> us_jacobian_column;     // buffer for holding jacobian column while differentiating by us

public:
    // This function must be called before any other function.
    void prepare(HandInput&& input) override;

    void calculate_objective(int times) override;
    void calculate_jacobian(int times) override;
    HandOutput output() override;

    ~TapenadeHand() { free_objective_input(); }

private:
    HandObjectiveData* convert_to_hand_objective_data(const HandInput& input);
    static Matrix convert_to_matrix(const LightMatrix<double>& mat);
    static Matrix_diff buildMatrixDiff(const LightMatrix<double>& mat);

    void free_objective_input();
    void calculate_jacobian_simple();
    void calculate_jacobian_complicated();
};

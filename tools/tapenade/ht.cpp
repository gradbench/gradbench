// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeHand.cpp

#include "gradbench/evals/ht.hpp"
#include "gradbench/main.hpp"

#include "evals/hand/hand_d.h"

// Data for ht objective converted from input
struct HTObjectiveData {
  int          bone_count;
  const char** bone_names;
  const int* parents;  // assumimng that parent is earlier in the order of bones
  Matrix*    base_relatives;
  Matrix*    inverse_base_absolutes;
  Matrix     base_positions;
  Matrix     weights;
  const Triangle* triangles;
  int             is_mirrored;
  int             corresp_count;
  const int*      correspondences;
  Matrix          points;
};

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  HTObjectiveData*    _objective_input = nullptr;
  bool                _complicated     = false;
  std::vector<double> _objective;

  std::vector<double> theta_d;  // buffer for theta differentiation directions
  std::vector<double> us_d;     // buffer for us differentiation directions
  Matrix_diff*        base_relativesd;  // buffer needed to hold the diff of
                                        // objective_input->base_relatives
  Matrix_diff*
      inverse_base_absolutesd;  // buffer needed to hold the diff of
                                // objective_input->inverse_base_absolutes
  Matrix_diff base_positionsd;  // buffer needed to hold the diff of
                                // objective_input->base_positions

  std::vector<double> us_jacobian_column;  // buffer for holding jacobian column
                                           // while differentiating by us

  HTObjectiveData*   convert_to_ht_objective_data(const ht::Input& input);
  static Matrix      convert_to_matrix(const LightMatrix<double>& mat);
  static Matrix_diff buildMatrixDiff(const LightMatrix<double>& mat);

  void free_objective_input();
  void calculate_jacobian_simple(ht::JacOutput&);
  void calculate_jacobian_complicated(ht::JacOutput&);

public:
  Jacobian(ht::Input& input)
      : Function(input), _complicated(input.us.size() != 0),
        _objective(3 * input.data.correspondences.size()) {
    if (_objective_input != nullptr) {
      free_objective_input();
    }

    this->_objective_input = convert_to_ht_objective_data(_input);

    int err_size = 3 * _input.data.correspondences.size();
    int ncols    = _input.theta.size();
    if (_complicated) {
      ncols += 2;
    }

    theta_d            = std::vector<double>(_input.theta.size());
    us_d               = std::vector<double>(_input.us.size());
    us_jacobian_column = std::vector<double>(err_size);
  }

  void compute(ht::JacOutput& output) {
    int err_size          = 3 * _input.data.correspondences.size();
    int ncols             = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (_complicated) {
      calculate_jacobian_complicated(output);
    } else {
      calculate_jacobian_simple(output);
    }
  }
};

void Jacobian::calculate_jacobian_simple(ht::JacOutput& output) {
  for (size_t i = 0; i < theta_d.size(); i++) {
    if (i > 0) {
      theta_d[i - 1] = 0.0;
    }

    theta_d[i] = 1.0;
    hand_objective_d(
        _input.theta.data(), theta_d.data(), _objective_input->bone_count,
        _objective_input->bone_names, _objective_input->parents,
        _objective_input->base_relatives, base_relativesd,
        _objective_input->inverse_base_absolutes, inverse_base_absolutesd,
        &_objective_input->base_positions, &base_positionsd,
        &_objective_input->weights, _objective_input->triangles,
        _objective_input->is_mirrored, _objective_input->corresp_count,
        _objective_input->correspondences, &_objective_input->points,
        _objective.data(), output.jacobian.data() + i * output.jacobian_nrows);
  }

  theta_d.back() = 0.0;
}

void Jacobian::calculate_jacobian_complicated(ht::JacOutput& output) {
  int nrows = _objective.size();
  int shift = 2 * nrows;

  // calculate theta jacobian part
  for (size_t i = 0; i < theta_d.size(); i++) {
    if (i > 0) {
      theta_d[i - 1] = 0.0;
    }

    theta_d[i] = 1.0;
    hand_objective_complicated_d(
        _input.theta.data(), theta_d.data(), _input.us.data(), us_d.data(),
        _objective_input->bone_count, _objective_input->bone_names,
        _objective_input->parents, _objective_input->base_relatives,
        base_relativesd, _objective_input->inverse_base_absolutes,
        inverse_base_absolutesd, &_objective_input->base_positions,
        &base_positionsd, &_objective_input->weights,
        _objective_input->triangles, _objective_input->is_mirrored,
        _objective_input->corresp_count, _objective_input->correspondences,
        &_objective_input->points, _objective.data(),
        output.jacobian.data() + shift + i * nrows);
  }

  theta_d.back() = 0.0;

  // calculate us jacobian part
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < us_d.size(); j++) {
      us_d[j] = (1 + i + j) % 2;
    }

    hand_objective_complicated_d(
        _input.theta.data(), theta_d.data(), _input.us.data(), us_d.data(),
        _objective_input->bone_count, _objective_input->bone_names,
        _objective_input->parents, _objective_input->base_relatives,
        base_relativesd, _objective_input->inverse_base_absolutes,
        inverse_base_absolutesd, &_objective_input->base_positions,
        &base_positionsd, &_objective_input->weights,
        _objective_input->triangles, _objective_input->is_mirrored,
        _objective_input->corresp_count, _objective_input->correspondences,
        &_objective_input->points, _objective.data(),
        us_jacobian_column.data());

    int Jrows = 3 * _input.data.correspondences.size();
    for (size_t j = 0; j < us_jacobian_column.size(); j++) {
      (output.jacobian)[i * Jrows + j] = us_jacobian_column[j];
    }

    for (size_t j = 0; j < us_d.size(); j++) {
      us_d[j] = 0;
    }
  }

  us_d.back() = 0.0;
}

HTObjectiveData*
Jacobian::convert_to_ht_objective_data(const ht::Input& input) {
  HTObjectiveData* _output = new HTObjectiveData;

  _output->correspondences = input.data.correspondences.data();
  _output->corresp_count   = input.data.correspondences.size();
  _output->points          = convert_to_matrix(input.data.points);

  const ht::ModelLightMatrix& imd = input.data.model;
  _output->bone_count             = imd.bone_names.size();
  _output->parents                = imd.parents.data();
  _output->base_positions         = convert_to_matrix(imd.base_positions);
  base_positionsd                 = buildMatrixDiff(imd.base_positions);
  _output->weights                = convert_to_matrix(imd.weights);
  _output->triangles              = imd.triangles.data();
  _output->is_mirrored            = imd.is_mirrored ? 1 : 0;

  _output->bone_names             = new const char*[_output->bone_count];
  _output->base_relatives         = new Matrix[_output->bone_count];
  base_relativesd                 = new Matrix_diff[_output->bone_count];
  _output->inverse_base_absolutes = new Matrix[_output->bone_count];
  inverse_base_absolutesd         = new Matrix_diff[_output->bone_count];

  for (int i = 0; i < _output->bone_count; i++) {
    _output->bone_names[i]     = imd.bone_names[i].data();
    _output->base_relatives[i] = convert_to_matrix(imd.base_relatives[i]);
    base_relativesd[i]         = buildMatrixDiff(imd.base_relatives[i]);
    _output->inverse_base_absolutes[i] =
        convert_to_matrix(imd.inverse_base_absolutes[i]);
    inverse_base_absolutesd[i] = buildMatrixDiff(imd.inverse_base_absolutes[i]);
  }

  return _output;
}

Matrix Jacobian::convert_to_matrix(const LightMatrix<double>& mat) {
  return {mat.nrows_, mat.ncols_, mat.data_};
}

Matrix_diff Jacobian::buildMatrixDiff(const LightMatrix<double>& mat) {
  int     length       = mat.nrows_ * mat.ncols_;
  double* dataContents = (double*)malloc(length * sizeof(double));
  for (int j = 0; j < length; ++j)
    dataContents[j] = 0.0;
  return {dataContents};
}

void Jacobian::free_objective_input() {
  if (_objective_input != nullptr) {
    free(base_positionsd.data);
    for (int i = 0; i < _objective_input->bone_count; i++) {
      free(base_relativesd[i].data);
      free(inverse_base_absolutesd[i].data);
    }
    delete[] _objective_input->bone_names;
    delete[] _objective_input->base_relatives;
    delete[] base_relativesd;
    delete[] _objective_input->inverse_base_absolutes;
    delete[] inverse_base_absolutesd;

    delete _objective_input;
    _objective_input = nullptr;
  }
}

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}

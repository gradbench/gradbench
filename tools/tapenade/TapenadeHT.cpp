// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeHand.cpp

#include "TapenadeHT.h"

TapenadeHand::TapenadeHand(HandInput& input) : ITest(input) {
  complicated = _input.us.size() > 0;

  if (objective_input != nullptr)
    {
      free_objective_input();
    }

  this->objective_input = convert_to_hand_objective_data(_input);

  int err_size = 3 * _input.data.correspondences.size();
  int ncols = _input.theta.size();
  if (complicated)
    {
      ncols += 2;
    }

  _output = {
    std::vector<double>(err_size),
    ncols,
    err_size,
    std::vector<double>(err_size * ncols)
  };

  theta_d = std::vector<double>(_input.theta.size());
  us_d = std::vector<double>(_input.us.size());
  us_jacobian_column = std::vector<double>(err_size);
}

void TapenadeHand::calculate_objective() {
  if (complicated) {
    hand_objective_complicated(_input.theta.data(),
                               _input.us.data(),
                               objective_input->bone_count,
                               objective_input->bone_names,
                               objective_input->parents,
                               objective_input->base_relatives,
                               objective_input->inverse_base_absolutes,
                               &objective_input->base_positions,
                               &objective_input->weights,
                               objective_input->triangles,
                               objective_input->is_mirrored,
                               objective_input->corresp_count,
                               objective_input->correspondences,
                               &objective_input->points,
                               _output.objective.data()
                               );
  } else {
    hand_objective(_input.theta.data(),
                   objective_input->bone_count,
                   objective_input->bone_names,
                   objective_input->parents,
                   objective_input->base_relatives,
                   objective_input->inverse_base_absolutes,
                   &objective_input->base_positions,
                   &objective_input->weights,
                   objective_input->triangles,
                   objective_input->is_mirrored,
                   objective_input->corresp_count,
                   objective_input->correspondences,
                   &objective_input->points,
                   _output.objective.data()
                   );
  }
}

void TapenadeHand::calculate_jacobian()
{
  if (complicated) {
    calculate_jacobian_complicated();
  } else {
    calculate_jacobian_simple();
  }
}

void TapenadeHand::calculate_jacobian_simple()
{
    for (size_t i = 0; i < theta_d.size(); i++)
    {
        if (i > 0)
        {
            theta_d[i - 1] = 0.0;
        }

        theta_d[i] = 1.0;
        hand_objective_d(
            _input.theta.data(),
            theta_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            base_relativesd,
            objective_input->inverse_base_absolutes,
            inverse_base_absolutesd,
            &objective_input->base_positions,
            &base_positionsd,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            _output.objective.data(),
            _output.jacobian.data() + i * _output.jacobian_nrows
        );
    }

    theta_d.back() = 0.0;
}



void TapenadeHand::calculate_jacobian_complicated()
{
    int nrows = _output.objective.size();
    int shift = 2 * nrows;

    // calculate theta jacobian part
    for (size_t i = 0; i < theta_d.size(); i++)
    {
        if (i > 0)
        {
            theta_d[i - 1] = 0.0;
        }

        theta_d[i] = 1.0;
        hand_objective_complicated_d(
            _input.theta.data(),
            theta_d.data(),
            _input.us.data(),
            us_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            base_relativesd,
            objective_input->inverse_base_absolutes,
            inverse_base_absolutesd,
            &objective_input->base_positions,
            &base_positionsd,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            _output.objective.data(),
            _output.jacobian.data() + shift + i * nrows
        );
    }

    theta_d.back() = 0.0;

    // calculate us jacobian part
    for (size_t i = 0; i < us_d.size(); i++)
    {
        if (i > 0)
        {
            us_d[i - 1] = 0.0;
        }

        us_d[i] = 1.0;
        hand_objective_complicated_d(
            _input.theta.data(),
            theta_d.data(),
            _input.us.data(),
            us_d.data(),
            objective_input->bone_count,
            objective_input->bone_names,
            objective_input->parents,
            objective_input->base_relatives,
            base_relativesd,
            objective_input->inverse_base_absolutes,
            inverse_base_absolutesd,
            &objective_input->base_positions,
            &base_positionsd,
            &objective_input->weights,
            objective_input->triangles,
            objective_input->is_mirrored,
            objective_input->corresp_count,
            objective_input->correspondences,
            &objective_input->points,
            _output.objective.data(),
            us_jacobian_column.data()
        );

        if (i % 2 == 0)
        {
            for (int j = 0; j < 3; j++)
            {
                _output.jacobian[3 * (i / 2) + j] = us_jacobian_column[3 * (i / 2) + j];
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                _output.jacobian[nrows + 3 * ((i - 1) / 2) + j] = us_jacobian_column[3 * ((i - 1) / 2) + j];
            }
        }
    }

    us_d.back() = 0.0;
}



HandObjectiveData* TapenadeHand::convert_to_hand_objective_data(const HandInput& _input)
{
    HandObjectiveData* _output = new HandObjectiveData;

    _output->correspondences = _input.data.correspondences.data();
    _output->corresp_count = _input.data.correspondences.size();
    _output->points = convert_to_matrix(_input.data.points);

    const HandModelLightMatrix& imd = _input.data.model;
    _output->bone_count = imd.bone_names.size();
    _output->parents = imd.parents.data();
    _output->base_positions = convert_to_matrix(imd.base_positions);
    base_positionsd = buildMatrixDiff(imd.base_positions) ;
    _output->weights = convert_to_matrix(imd.weights);
    _output->triangles = imd.triangles.data();
    _output->is_mirrored = imd.is_mirrored ? 1 : 0;

    _output->bone_names = new const char* [_output->bone_count];
    _output->base_relatives = new Matrix[_output->bone_count];
    base_relativesd = new Matrix_diff[_output->bone_count];
    _output->inverse_base_absolutes = new Matrix[_output->bone_count];
    inverse_base_absolutesd = new Matrix_diff[_output->bone_count];

    for (int i = 0; i < _output->bone_count; i++)
    {
        _output->bone_names[i] = imd.bone_names[i].data();
        _output->base_relatives[i] = convert_to_matrix(imd.base_relatives[i]);
        base_relativesd[i] = buildMatrixDiff(imd.base_relatives[i]);
        _output->inverse_base_absolutes[i] = convert_to_matrix(imd.inverse_base_absolutes[i]);
        inverse_base_absolutesd[i] = buildMatrixDiff(imd.inverse_base_absolutes[i]);
    }

    return _output;
}



Matrix TapenadeHand::convert_to_matrix(const LightMatrix<double>& mat)
{
    return {
        mat.nrows_,
        mat.ncols_,
        mat.data_
    };
}

Matrix_diff TapenadeHand::buildMatrixDiff(const LightMatrix<double>& mat)
{
    int length = mat.nrows_ * mat.ncols_ ;
    double *dataContents = (double*)malloc(length * sizeof(double));
    for (int j=0 ; j<length ; ++j) dataContents[j] = 0.0 ;
    return {dataContents} ;
}

void TapenadeHand::free_objective_input()
{
    if (objective_input != nullptr)
    {
        free(base_positionsd.data) ;
        for (int i = 0; i < objective_input->bone_count; i++) {
          free(base_relativesd[i].data) ;
          free(inverse_base_absolutesd[i].data) ;
        }
        delete[] objective_input->bone_names;
        delete[] objective_input->base_relatives;
        delete[] base_relativesd;
        delete[] objective_input->inverse_base_absolutes;
        delete[] inverse_base_absolutesd;

        delete objective_input;
        objective_input = nullptr;
    }
}

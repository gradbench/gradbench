// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp

#include "gradbench/main.hpp"
#include "gradbench/evals/ht.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/taping.h>
#include <adolc/interfaces.h>

struct Pointer2 {
  Pointer2(int n, int m) : n(n), m(m) {
    data = new double*[n];
    for (int i = 0; i < n; i++)
      data[i] = new double[m];
  }
  ~Pointer2() {
    for (int i = 0; i < n; i++)
      delete[] data[i];
    delete[] data;
  }
  double*& operator[](int i) { return data[i]; }
  double **data;
  int n;
  int m;
};

void compute_ht_complicated_J(const std::vector<double>& theta, const std::vector<double>& us,
                              const ht::DataLightMatrix& data,
                              std::vector<double> *perr, std::vector<double> *pJ) {
  auto &err = *perr;
  auto &J = *pJ;

  int tapeTag = 1;
  int Jrows = (int)err.size();
  int n_independents = (int)(us.size() + theta.size());
  size_t n_pts = err.size() / 3;
  int ndirs = 2 + (int)theta.size();
  std::vector<adouble> aus(us.size());
  std::vector<adouble> atheta(theta.size());
  std::vector<adouble> aerr(err.size());

  std::vector<double> all_params(n_independents);
  for (size_t i = 0; i < us.size(); i++)
    all_params[i] = us[i];
  for (size_t i = 0; i < theta.size(); i++)
    all_params[i + us.size()] = theta[i];

  // create seed matrix
  Pointer2 seed(n_independents, ndirs);
  for (int i = 0; i < n_independents; i++)
    std::fill(seed[i], seed[i] + ndirs, (double)0);
  for (size_t i = 0; i < n_pts; i++) {
    seed[2 * i][0] = 1.;
    seed[2 * i + 1][1] = 1.;
  }
  for (size_t i = 0; i < theta.size(); i++)
    seed[us.size() + i][2 + i] = 1.;

  Pointer2 J_tmp(Jrows, ndirs);

  // Record on a tape
  trace_on(tapeTag);
  for (size_t i = 0; i < us.size(); i++)
    aus[i] <<= us[i];
  for (size_t i = 0; i < theta.size(); i++)
    atheta[i] <<= theta[i];

  ht::objective(&atheta[0], &aus[0], &data, &aerr[0]);

  for (int i = 0; i < Jrows; i++)
    aerr[i] >>= err[i];

  trace_off();

  fov_forward(tapeTag, Jrows, n_independents, ndirs, &all_params[0], seed.data, &err[0], J_tmp.data);

  for (int i = 0; i < Jrows; i++)
    for (int j = 0; j < ndirs; j++)
      J[j*Jrows + i] = J_tmp[i][j];
}

void get_ht_nnz_pattern(const ht::DataLightMatrix& data,
                        std::vector<unsigned int> *pridxs,
                        std::vector<unsigned int> *pcidxs,
                        std::vector<double> *pnzvals)
{
  auto& ridxs = *pridxs;
  auto& cidxs = *pcidxs;

  int n_pts = (int)data.points.cols();
  int nnz_estimate = 3 * n_pts*(3 + 1 + 4);
  ridxs.reserve(nnz_estimate); cidxs.reserve(nnz_estimate);

  for (int i = 0; i < 3*n_pts; i++)
    {
      for (int j = 0; j < 3; j++)
        {
          ridxs.push_back(i);
          cidxs.push_back(j);
        }
      ridxs.push_back(i);
      cidxs.push_back(3 + (i % 3));
    }
  int col_off = 6;

  const auto& parents = data.model.parents;
  for (int i_pt = 0; i_pt < n_pts; i_pt++)
    {
      int i_vert = data.correspondences[i_pt];
      std::vector<bool> bones(data.model.bone_names.size(), false);
      for (size_t i_bone = 0; i_bone < bones.size(); i_bone++)
        {
          bones[i_bone] = bones[i_bone] | (data.model.weights(i_bone, i_vert) != 0);
        }
      for (int i_bone = (int)parents.size()-1; i_bone >= 0; i_bone--)
        {
          if(parents[i_bone] >= 0)
            bones[parents[i_bone]] = bones[i_bone] | bones[parents[i_bone]];
        }
      int i_col = col_off;
      for (int i_finger = 0; i_finger < 5; i_finger++)
        {
          for (int i_finger_bone = 1; i_finger_bone < 4; i_finger_bone++)
            {
              int i_bone = 1 + i_finger * 4 + i_finger_bone;
              if (bones[i_bone])
                {
                  for (int i_coord = 0; i_coord < 3; i_coord++)
                    {
                      ridxs.push_back(i_pt*3 + i_coord);
                      cidxs.push_back(i_col);
                    }
                }
              i_col++;
              if (i_finger_bone == 1)
                {
                  if (bones[i_bone])
                    {
                      for (int i_coord = 0; i_coord < 3; i_coord++)
                        {
                          ridxs.push_back(i_pt * 3 + i_coord);
                          cidxs.push_back(i_col);
                        }
                    }
                  i_col++;
                }
            }
        }
    }

  pnzvals->resize(cidxs.size());
}

void get_ht_nnz_pattern(int n_rows,
                        const std::vector<unsigned int>& ridxs,
                        const std::vector<unsigned int>& cidxs,
                        unsigned int ***ppattern)
{
  auto &pattern = *ppattern;

  std::vector<int> cols_counts(n_rows, 0);
  for (size_t i = 0; i < ridxs.size(); i++)
    cols_counts[ridxs[i]]++;

  pattern = new unsigned int*[n_rows];
  for (int i = 0; i < n_rows; i++) {
    pattern[i] = new unsigned int[cols_counts[i] + 1];
    pattern[i][0] = cols_counts[i];
  }

  std::vector<int> tails(n_rows, 1);
  for (size_t i = 0; i < ridxs.size(); i++) {
    pattern[ridxs[i]][tails[ridxs[i]]++] = cidxs[i];
  }
}

void compute_ht_simple_J(std::vector<double>& theta,
                         const ht::DataLightMatrix& data,
                         std::vector<double> *perr,
                         std::vector<double> *pJ)
{
  auto& err = *perr;
  auto& J = *pJ;

  int tapeTag = 1;
  int Jrows = 3* (int)data.correspondences.size();
  int Jcols = (int)theta.size();
  std::vector<adouble> atheta(Jcols);
  std::vector<adouble> aerr(Jrows);

  double **J_tmp = new double*[err.size()];
  for (size_t i = 0; i < err.size(); i++)
    J_tmp[i] = new double[theta.size()];

  // Record on a tape
  trace_on(tapeTag);

  for (size_t i = 0; i < theta.size(); i++)
    atheta[i] <<= theta[i];

  ht::objective(&atheta[0], &data, &aerr[0]);

  for (int i = 0; i < Jrows; i++)
    aerr[i] >>= err[i];

  trace_off();

  jacobian(tapeTag, Jrows, Jcols, &theta[0], J_tmp);

  for (int i = 0; i < Jrows; i++) {
    for (int j = 0; j < Jcols; j++)
      J[j*Jrows + i] = J_tmp[i][j];
    delete[] J_tmp[i];
  }
  delete[] J_tmp;
}

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool _complicated = false;
  std::vector<double> _objective;
public:
  Jacobian(ht::Input& input) :
    Function(input),
    _complicated(input.us.size() != 0),
    _objective(3 * input.data.correspondences.size())
  {}

  void compute(ht::JacOutput& output) {
    int err_size = 3 * _input.data.correspondences.size();
    int ncols = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (_complicated) {
      compute_ht_complicated_J(_input.theta, _input.us, _input.data, &_objective, &output.jacobian);
    } else {
      compute_ht_simple_J(_input.theta, _input.data, &_objective, &output.jacobian);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ht::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}

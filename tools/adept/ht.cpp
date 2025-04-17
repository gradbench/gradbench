// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Adept/main.cpp

#include "gradbench/evals/ht.hpp"
#include "adept.h"
#include "gradbench/main.hpp"

using adept::adouble;

void set_gradients(double val, std::vector<adouble>* aparams) {
  for (size_t i = 0; i < aparams->size(); i++)
    (*aparams)[i].set_gradient(val);
}

void compute_ht_complicated_J(const std::vector<double>& theta,
                              const std::vector<double>& us,
                              const ht::DataLightMatrix* data,
                              std::vector<double>*       perr,
                              std::vector<double>*       pJ) {
  auto&                err = *perr;
  auto&                J   = *pJ;
  adept::Stack         stack;
  std::vector<adouble> atheta(theta.size());
  std::vector<adouble> aus(us.size());
  std::vector<adouble> aerr(err.size());
  size_t               n_pts = us.size() / 2;

  adept::set_values(atheta.data(), theta.size(), theta.data());
  adept::set_values(aus.data(), us.size(), us.data());

  stack.new_recording();

  ht::objective(atheta.data(), aus.data(), data, aerr.data());
  adept::get_values(&aerr[0], err.size(), &err[0]);

  // Compute wrt. us
  set_gradients(0., &atheta);
  for (size_t i = 0; i < n_pts; i++) {
    aus[2 * i].set_gradient(1.);
    aus[2 * i + 1].set_gradient(0.);
  }
  stack.forward();
  adept::get_gradients(&aerr[0], aerr.size(), &J[0]);
  for (size_t i = 0; i < n_pts; i++) {
    aus[2 * i].set_gradient(0.);
    aus[2 * i + 1].set_gradient(1.);
  }
  stack.forward();
  adept::get_gradients(&aerr[0], aerr.size(), &J[aerr.size()]);
  for (size_t i = 0; i < n_pts; i++)
    aus[2 * i + 1].set_gradient(0.);
  int offset = 2;

  // Compute wrt. theta
  for (size_t i_param = 0; i_param < theta.size(); i_param++) {
    atheta[i_param].set_gradient(1.);
    stack.forward();
    adept::get_gradients(&aerr[0], aerr.size(),
                         &J[(offset + i_param) * aerr.size()]);
    atheta[i_param].set_gradient(0.);
  }
}

void compute_ht_simple_J(const std::vector<double>& theta,
                         const ht::DataLightMatrix* data,
                         std::vector<double>* perr, std::vector<double>* pJ) {
  auto&                err = *perr;
  auto&                J   = *pJ;
  adept::Stack         stack;
  std::vector<adouble> atheta(theta.size());
  std::vector<adouble> aerr(err.size());

  adept::set_values(atheta.data(), theta.size(), theta.data());

  stack.new_recording();
  ht::objective(atheta.data(), data, aerr.data());
  stack.independent(atheta.data(), atheta.size());
  stack.dependent(aerr.data(), aerr.size());
  stack.jacobian_forward(J.data());
  adept::get_values(aerr.data(), err.size(), err.data());
}

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool                _complicated = false;
  std::vector<double> _objective;

public:
  Jacobian(ht::Input& input)
      : Function(input), _complicated(input.us.size() != 0),
        _objective(3 * input.data.correspondences.size()) {}

  void compute(ht::JacOutput& output) {
    int err_size          = 3 * _input.data.correspondences.size();
    int ncols             = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (!_complicated) {
      compute_ht_simple_J(_input.theta, &_input.data, &_objective,
                          &output.jacobian);
    } else {
      compute_ht_complicated_J(_input.theta, _input.us, &_input.data,
                               &_objective, &output.jacobian);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}

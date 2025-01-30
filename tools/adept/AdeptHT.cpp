// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Adept/main.cpp

#include "AdeptHT.h"

#include "adbench/shared/ht_light_matrix.h"
#include "adept.h"

using adept::adouble;

void set_gradients(double val, vector<adouble> *aparams) {
  for (size_t i = 0; i < aparams->size(); i++)
    (*aparams)[i].set_gradient(val);
}

void compute_hand_complicated_J(const vector<double>& theta, const vector<double>& us,
                                const HandDataLightMatrix* data,
                                vector<double> *perr, vector<double> *pJ) {
  auto &err = *perr;
  auto &J = *pJ;
  adept::Stack stack;
  vector<adouble> atheta(theta.size());
  vector<adouble> aus(us.size());
  vector<adouble> aerr(err.size());
  size_t n_pts = us.size() / 2;

  adept::set_values(atheta.data(), theta.size(), theta.data());
  adept::set_values(aus.data(), us.size(), us.data());

  stack.new_recording();

  hand_objective(atheta.data(), aus.data(), data, aerr.data());
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
    adept::get_gradients(&aerr[0], aerr.size(), &J[(offset + i_param)*aerr.size()]);
    atheta[i_param].set_gradient(0.);
  }
}

void compute_hand_simple_J(const vector<double>& theta, const HandDataLightMatrix* data,
                           vector<double> *perr, vector<double> *pJ) {
  auto &err = *perr;
  auto &J = *pJ;
  adept::Stack stack;
  vector<adouble> atheta(theta.size());
  vector<adouble> aerr(err.size());

  adept::set_values(atheta.data(), theta.size(), theta.data());

  stack.new_recording();
  hand_objective(atheta.data(), data, aerr.data());
  stack.independent(atheta.data(), atheta.size());
  stack.dependent(aerr.data(), aerr.size());
  stack.jacobian_forward(J.data());
  adept::get_values(aerr.data(), err.size(), err.data());
}

AdeptHand::AdeptHand(HandInput& input) : ITest(input) {
  _complicated = _input.us.size() != 0;
  int err_size = 3 * _input.data.correspondences.size();
  int ncols = (_complicated ? 2 : 0) + _input.theta.size();
  _output = { std::vector<double>(err_size), ncols, err_size, std::vector<double>(err_size * ncols) };
}

void AdeptHand::calculate_objective() {
  if (_complicated) {
    hand_objective(_input.theta.data(), _input.us.data(), &_input.data, _output.objective.data());
  } else {
    hand_objective(_input.theta.data(), &_input.data, _output.objective.data());
  }
}

void AdeptHand::calculate_jacobian() {
  if (_input.us.size() == 0) {
    compute_hand_simple_J(_input.theta,
                          &_input.data,
                          &_output.objective,
                          &_output.jacobian);
  } else {
    compute_hand_complicated_J(_input.theta,
                               _input.us,
                               &_input.data,
                               &_output.objective,
                               &_output.jacobian);
  }
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "AdeptLSTM.h"
#include "adbench/shared/lstm.h"
#include "adept.h"

using adept::adouble;

void lstm_objective_J(int l, int c, int b,
                      const LSTMInput& input,
                      double* loss, double* J) {
  adept::Stack stack;
  std::vector<adouble> amain_params(input.main_params.size());
  std::vector<adouble> aextra_params(input.extra_params.size());
  std::vector<adouble> astate(input.state.size());
  std::vector<adouble> asequence(input.sequence.size());

  adept::set_values(amain_params.data(), input.main_params.size(), input.main_params.data());
  adept::set_values(aextra_params.data(), input.extra_params.size(), input.extra_params.data());
  adept::set_values(astate.data(), input.state.size(), input.state.data());
  adept::set_values(asequence.data(), input.sequence.size(), input.sequence.data());

  stack.new_recording();
  adouble aloss;
  lstm_objective(l, c, b,
                 amain_params.data(), aextra_params.data(),
                 astate, asequence.data(),
                 &aloss);
  aloss.set_gradient(1.); // only one J row here
  stack.reverse();

  int offset = 0;

  adept::get_gradients(amain_params.data(), input.main_params.size(), J+offset);
  offset += input.main_params.size();

  adept::get_gradients(aextra_params.data(), input.extra_params.size(), J+offset);
  offset += input.extra_params.size();
}


void AdeptLSTM::prepare(LSTMInput&& input) {
  this->input = input;
  int Jcols = 8 * this->input.l * this->input.b + 3 * this->input.b;
  state = std::vector<double>(this->input.state.size());
  result = { 0, std::vector<double>(Jcols) };
}

LSTMOutput AdeptLSTM::output()
{
  return result;
}

void AdeptLSTM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    state = input.state;
    lstm_objective(input.l, input.c, input.b, input.main_params.data(), input.extra_params.data(), state, input.sequence.data(), &result.objective);
  }
}

void AdeptLSTM::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    state = input.state;
    lstm_objective_J(input.l, input.c, input.b, input, &result.objective, result.gradient.data());
  }
}

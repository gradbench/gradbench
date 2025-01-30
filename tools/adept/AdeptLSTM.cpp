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
                 astate.data(), asequence.data(),
                 &aloss);
  aloss.set_gradient(1.); // only one J row here
  stack.reverse();

  int offset = 0;

  adept::get_gradients(amain_params.data(), input.main_params.size(), J+offset);
  offset += input.main_params.size();

  adept::get_gradients(aextra_params.data(), input.extra_params.size(), J+offset);
  offset += input.extra_params.size();
}


AdeptLSTM::AdeptLSTM(LSTMInput& input) : ITest(input) {
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
}

void AdeptLSTM::calculate_objective() {
  lstm_objective(_input.l, _input.c, _input.b, _input.main_params.data(), _input.extra_params.data(), _input.state.data(), _input.sequence.data(), &_output.objective);
}

void AdeptLSTM::calculate_jacobian() {
  lstm_objective_J(_input.l, _input.c, _input.b, _input, &_output.objective, _output.gradient.data());
}

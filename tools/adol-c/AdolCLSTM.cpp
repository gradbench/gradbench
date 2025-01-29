#include "AdolCLSTM.h"
#include "adbench/shared/lstm.h"

#include <string.h>


#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>


static const int tapeTag = 1;

void AdolCLSTM::prepare(LSTMInput&& input) {
  this->input = input;
  int Jcols = 8 * this->input.l * this->input.b + 3 * this->input.b;
  result = { 0, std::vector<double>(Jcols) };

  // Construct tape.

  trace_on(tapeTag);

  std::vector<adouble> amain_params(input.main_params.size());
  std::vector<adouble> aextra_params(input.extra_params.size());
  std::vector<adouble> astate(input.state.size());
  std::vector<adouble> asequence(input.sequence.size());
  adouble aobjective;

  for (size_t i = 0; i < input.main_params.size(); i++) {
    amain_params[i] <<= input.main_params[i];
  }

  for (size_t i = 0; i < input.extra_params.size(); i++) {
    aextra_params[i] <<= input.extra_params[i];
  }

  for (size_t i = 0; i < input.state.size(); i++) {
    astate[i] = input.state[i];
  }

  for (size_t i = 0; i < input.sequence.size(); i++) {
    asequence[i] = input.sequence[i];
  }

  lstm_objective(input.l, input.c, input.b,
                 amain_params.data(), aextra_params.data(),
                 astate.data(), asequence.data(),
                 &aobjective);

  aobjective >>= result.objective;

  trace_off();
}

LSTMOutput AdolCLSTM::output() {
  return result;
}

void AdolCLSTM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    lstm_objective(input.l, input.c, input.b,
                   input.main_params.data(), input.extra_params.data(),
                   input.state.data(), input.sequence.data(),
                   &result.objective);
  }
}

void AdolCLSTM::calculate_jacobian(int times) {
  int Jcols = input.main_params.size() + input.extra_params.size();

  for (int i = 0; i < times; ++i) {
    double *in = new double[Jcols];
    memcpy(in,
           input.main_params.data(),
           input.main_params.size() * sizeof(double));
    memcpy(in + input.main_params.size(),
           input.extra_params.data(),
           input.extra_params.size() * sizeof(double));
    gradient(tapeTag, Jcols, in, result.gradient.data());
    delete[] in;
  }
}

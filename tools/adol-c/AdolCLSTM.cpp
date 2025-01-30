#include "AdolCLSTM.h"
#include "adbench/shared/lstm.h"

#include <string.h>


#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>


static const int tapeTag = 1;

AdolCLSTM::AdolCLSTM(LSTMInput& input) : ITest(input) {
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
}

void AdolCLSTM::prepare_jacobian() {
  trace_on(tapeTag);

  std::vector<adouble> amain_params(_input.main_params.size());
  std::vector<adouble> aextra_params(_input.extra_params.size());
  std::vector<adouble> astate(_input.state.size());
  std::vector<adouble> asequence(_input.sequence.size());
  adouble aobjective;

  for (size_t i = 0; i < _input.main_params.size(); i++) {
    amain_params[i] <<= _input.main_params[i];
  }

  for (size_t i = 0; i < _input.extra_params.size(); i++) {
    aextra_params[i] <<= _input.extra_params[i];
  }

  for (size_t i = 0; i < _input.state.size(); i++) {
    astate[i] = _input.state[i];
  }

  for (size_t i = 0; i < _input.sequence.size(); i++) {
    asequence[i] = _input.sequence[i];
  }

  lstm_objective(_input.l, _input.c, _input.b,
                 amain_params.data(), aextra_params.data(),
                 astate.data(), asequence.data(),
                 &aobjective);

  aobjective >>= _output.objective;

  trace_off();

}

void AdolCLSTM::calculate_objective() {
  lstm_objective(_input.l, _input.c, _input.b,
                 _input.main_params.data(), _input.extra_params.data(),
                 _input.state.data(), _input.sequence.data(),
                 &_output.objective);
}

void AdolCLSTM::calculate_jacobian() {
  int Jcols = _input.main_params.size() + _input.extra_params.size();

  double *in = new double[Jcols];
  memcpy(in,
         _input.main_params.data(),
         _input.main_params.size() * sizeof(double));
  memcpy(in + _input.main_params.size(),
         _input.extra_params.data(),
         _input.extra_params.size() * sizeof(double));
  gradient(tapeTag, Jcols, in, _output.gradient.data());
  delete[] in;
}

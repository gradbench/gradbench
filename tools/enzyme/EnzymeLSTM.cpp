#include "EnzymeLSTM.h"
#include "adbench/shared/lstm.h"

EnzymeLSTM::EnzymeLSTM(LSTMInput& input) : ITest(input) {
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
}

void EnzymeLSTM::calculate_objective() {
  lstm_objective(_input.l, _input.c, _input.b,
                 _input.main_params.data(),
                 _input.extra_params.data(),
                 _input.state.data(),
                 _input.sequence.data(),
                 &_output.objective);
}

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;

void EnzymeLSTM::calculate_jacobian() {
  std::fill(_output.gradient.begin(), _output.gradient.end(), 0);
  double d_err = 1;
  double *d_main_params = _output.gradient.data();
  double *d_extra_params = _output.gradient.data() + _input.main_params.size();
  __enzyme_autodiff(lstm_objective<double>,
                    enzyme_const, _input.l,
                    enzyme_const, _input.c,
                    enzyme_const, _input.b,

                    enzyme_dup, _input.main_params.data(), d_main_params,
                    enzyme_dup, _input.extra_params.data(), d_extra_params,

                    enzyme_const, _input.state.data(),
                    enzyme_const, _input.sequence.data(),

                    enzyme_dupnoneed, &_output.objective, &d_err);
}

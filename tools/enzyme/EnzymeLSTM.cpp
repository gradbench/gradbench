#include "EnzymeLSTM.h"
#include "adbench/shared/lstm.h"

void EnzymeLSTM::prepare(LSTMInput&& input) {
  this->input = input;
  int Jcols = 8 * this->input.l * this->input.b + 3 * this->input.b;
  result = { 0, std::vector<double>(Jcols) };
}

LSTMOutput EnzymeLSTM::output()
{
  return result;
}

void EnzymeLSTM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    lstm_objective(input.l, input.c, input.b,
                   input.main_params.data(),
                   input.extra_params.data(),
                   input.state.data(),
                   input.sequence.data(),
                   &result.objective);
  }
}

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;

void EnzymeLSTM::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    std::fill(result.gradient.begin(), result.gradient.end(), 0);
    double d_err = 1;
    double *d_main_params = result.gradient.data();
    double *d_extra_params = result.gradient.data() + input.main_params.size();
    __enzyme_autodiff(lstm_objective<double>,
                      enzyme_const, input.l,
                      enzyme_const, input.c,
                      enzyme_const, input.b,

                      enzyme_dup, input.main_params.data(), d_main_params,
                      enzyme_dup, input.extra_params.data(), d_extra_params,

                      enzyme_const, input.state.data(),
                      enzyme_const, input.sequence.data(),

                      enzyme_dupnoneed, &result.objective, &d_err);
  }
}

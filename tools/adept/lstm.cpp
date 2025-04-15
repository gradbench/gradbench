#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include "adept.h"

using adept::adouble;

void lstm_objective_J(int l, int c, int b,
                      const lstm::Input& input,
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
  lstm::objective(l, c, b,
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

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
public:
  Jacobian(lstm::Input& input) : Function(input) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    double err;
    lstm_objective_J(_input.l, _input.c, _input.b,
                     _input,
                     &err,
                     output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}

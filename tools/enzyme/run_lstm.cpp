#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include "enzyme.h"

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
public:
  Jacobian(lstm::Input& input) : Function(input) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    std::fill(output.begin(), output.end(), 0);
    double err;
    double d_err = 1;
    double *d_main_params = output.data();
    double *d_extra_params = output.data() + _input.main_params.size();
    __enzyme_autodiff(lstm::objective<double>,
                      enzyme_const, _input.l,
                      enzyme_const, _input.c,
                      enzyme_const, _input.b,

                      enzyme_dup, _input.main_params.data(), d_main_params,
                      enzyme_dup, _input.extra_params.data(), d_extra_params,

                      enzyme_const, _input.state.data(),
                      enzyme_const, _input.sequence.data(),

                      enzyme_dupnoneed, &err, &d_err);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}

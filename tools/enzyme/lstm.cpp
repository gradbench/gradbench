#include "gradbench/evals/lstm.hpp"
#include "enzyme.h"
#include "gradbench/main.hpp"
#include <algorithm>

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
private:
  std::vector<double> _sequence_copy;

public:
  Jacobian(lstm::Input& input)
      : Function(input), _sequence_copy(input.sequence.size()) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    std::fill(output.begin(), output.end(), 0);
    double  err;
    double  d_err          = 1;
    double* d_main_params  = output.data();
    double* d_extra_params = output.data() + _input.main_params.size();

    // XXX: there seems to be a bug in either Enzyme or our use of it,
    // that causes it to slightly modify the sequence argument. If we
    // use it many times (i.e., with many runs), we will get wrong
    // results. As a workaround, do a copy so we have a fresh set of
    // values for every run.
    _sequence_copy = _input.sequence;

    __enzyme_autodiff(lstm::objective<double>, enzyme_const, _input.l,
                      enzyme_const, _input.c, enzyme_const, _input.b,

                      enzyme_dup, _input.main_params.data(), d_main_params,
                      enzyme_dup, _input.extra_params.data(), d_extra_params,

                      enzyme_const, _input.state.data(), enzyme_const,
                      _sequence_copy.data(),

                      enzyme_dupnoneed, &err, &d_err);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}

#include <algorithm>

#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
public:
  Jacobian(lstm::Input& input) : Function(input) {
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

    lstm::objective(_input.l, _input.c, _input.b,
                    amain_params.data(), aextra_params.data(),
                    astate.data(), asequence.data(),
                    &aobjective);

    double err;
    aobjective >>= err;

    trace_off();
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    double *in = new double[output.size()];
    memcpy(in,
           _input.main_params.data(),
           _input.main_params.size() * sizeof(double));
    memcpy(in + _input.main_params.size(),
           _input.extra_params.data(),
           _input.extra_params.size() * sizeof(double));
    gradient(tapeTag, output.size(), in, output.data());
    delete[] in;
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}

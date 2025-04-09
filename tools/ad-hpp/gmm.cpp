#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"
#include "ad.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000; // Determined experimentally.

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
    std::vector<adjoint_t> _alphas;
    std::vector<adjoint_t> _means;
    std::vector<adjoint_t> _icf;

public:
  Jacobian(gmm::Input& input) :
    Function(input),
    _alphas(_input.k),
    _means(_input.d * _input.k),
    _icf((_input.d*(_input.d + 1) / 2)*_input.k) {
    for (size_t i = 0; i < _alphas.size(); i++) {
      _alphas[i] = _input.alphas[i];
    }

    for (size_t i = 0; i < _means.size(); i++) {
      _means[i] = _input.means[i];
    }

    for (size_t i = 0; i < _icf.size(); i++) {
      _icf[i] = _input.icf[i];
    }
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    adjoint::global_tape = adjoint::tape_t::create(TAPE_SIZE);

    for (size_t i = 0; i < _alphas.size(); i++) {
      adjoint::global_tape->register_variable(_alphas[i]);
    }

    for (size_t i = 0; i < _means.size(); i++) {
      adjoint::global_tape->register_variable(_means[i]);
    }

    for (size_t i = 0; i < _icf.size(); i++) {
      adjoint::global_tape->register_variable(_icf[i]);
    }

    adjoint_t y;

    gmm::objective(_input.d, _input.k, _input.n,
                   _alphas.data(),
                   _means.data(),
                   _icf.data(),
                   _input.x.data(),
                   _input.wishart,
                   &y);

    ad::derivative(y) = 1.0;
    adjoint::global_tape->interpret_adjoint();

    int o = 0;
    for (size_t i = 0; i < _alphas.size(); i++) {
      output[o++] = ad::derivative(_alphas[i]);
    }
    for (size_t i = 0; i < _means.size(); i++) {
      output[o++] = ad::derivative(_means[i]);
    }
    for (size_t i = 0; i < _icf.size(); i++) {
      output[o++] = ad::derivative(_icf[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>},
    });;
}

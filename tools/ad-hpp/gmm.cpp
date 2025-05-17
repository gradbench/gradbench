#include <vector>

#include "ad.hpp"
#include "gradbench/evals/gmm.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000;  // Determined experimentally.

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  std::vector<adjoint_t> _alpha;
  std::vector<adjoint_t> _mu;
  std::vector<adjoint_t> _q;
  std::vector<adjoint_t> _l;

  ad::shared_global_tape_ptr<adjoint> _tape;

public:
  Jacobian(gmm::Input& input)
      : Function(input), _alpha(_input.k), _mu(_input.d * _input.k),
        _q(_input.k * _input.d), _l(_input.k * (_input.d * (_input.d - 1) / 2)),
        _tape(adjoint::tape_options_t(TAPE_SIZE)) {

    for (size_t i = 0; i < _alpha.size(); i++) {
      _alpha[i] = _input.alpha[i];
    }

    for (size_t i = 0; i < _mu.size(); i++) {
      _mu[i] = _input.mu[i];
    }

    for (size_t i = 0; i < _q.size(); i++) {
      _q[i] = _input.q[i];
    }

    for (size_t i = 0; i < _l.size(); i++) {
      _l[i] = _input.l[i];
    }
  }

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    _tape->reset();

    for (size_t i = 0; i < _alpha.size(); i++) {
      _tape->register_variable(_alpha[i]);
    }

    for (size_t i = 0; i < _mu.size(); i++) {
      _tape->register_variable(_mu[i]);
    }

    for (size_t i = 0; i < _q.size(); i++) {
      _tape->register_variable(_q[i]);
    }

    for (size_t i = 0; i < _l.size(); i++) {
      _tape->register_variable(_l[i]);
    }

    adjoint_t y;

    gmm::objective(_input.d, _input.k, _input.n, _alpha.data(), _mu.data(),
                   _q.data(), _l.data(), _input.x.data(), _input.wishart, &y);

    ad::derivative(y) = 1.0;
    _tape->interpret_adjoint();

    for (size_t i = 0; i < _alpha.size(); i++) {
      output.alpha[i] = ad::derivative(_alpha[i]);
    }
    for (size_t i = 0; i < _mu.size(); i++) {
      output.mu[i] = ad::derivative(_mu[i]);
    }
    for (size_t i = 0; i < _q.size(); i++) {
      output.q[i] = ad::derivative(_q[i]);
    }
    for (size_t i = 0; i < _l.size(); i++) {
      output.l[i] = ad::derivative(_l[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"objective", function_main<gmm::Objective>},
                          {"jacobian", function_main<Jacobian>},
                      });
  ;
}

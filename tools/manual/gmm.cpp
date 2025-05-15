#include "gradbench/evals/gmm.hpp"
#include "gmm_d.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    double error;
    gmm_objective_d(_input.d, _input.k, _input.n, _input.alphas.data(),
                    _input.mu.data(), _input.q.data(), _input.l.data(),
                    _input.x.data(), _input.wishart, &error,
                    output.alpha.data(), output.mu.data(), output.q.data(),
                    output.l.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}

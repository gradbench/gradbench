#include "gradbench/evals/gmm.hpp"
#include "enzyme.h"
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

    std::fill(output.alpha.begin(), output.alpha.end(), 0);
    std::fill(output.mu.begin(), output.mu.end(), 0);
    std::fill(output.q.begin(), output.q.end(), 0);
    std::fill(output.l.begin(), output.l.end(), 0);

    double* d_alpha = output.alpha.data();
    double* d_mu    = output.mu.data();
    double* d_q     = output.q.data();
    double* d_l     = output.l.data();

    double err;
    double d_err = 1;
    __enzyme_autodiff(gmm::objective<double>,

                      enzyme_const, _input.d,

                      enzyme_const, _input.k,

                      enzyme_const, _input.n,

                      enzyme_dup, _input.alpha.data(), d_alpha,

                      enzyme_dup, _input.mu.data(), d_mu,

                      enzyme_dup, _input.q.data(), d_q,

                      enzyme_dup, _input.l.data(), d_l,

                      enzyme_const, _input.x.data(),

                      enzyme_const, _input.wishart,

                      enzyme_dupnoneed, &err, &d_err);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}

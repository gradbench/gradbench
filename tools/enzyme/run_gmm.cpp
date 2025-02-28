#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"
#include "enzyme.h"

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);
    std::fill(output.begin(), output.end(), 0);

    double* d_alphas = output.data();
    double* d_means = d_alphas + _input.alphas.size();
    double* d_icf = d_means + _input.means.size();

    double err;
    double d_err = 1;
    __enzyme_autodiff
      (gmm::objective<double>,
       enzyme_const, _input.d,
       enzyme_const, _input.k,
       enzyme_const, _input.n,

       enzyme_dup, _input.alphas.data(), d_alphas,

       enzyme_dup, _input.means.data(), d_means,

       enzyme_dup, _input.icf.data(), d_icf,

       enzyme_const, _input.x.data(),
       enzyme_const, _input.wishart,
       enzyme_dupnoneed, &err, &d_err);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}

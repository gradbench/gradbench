#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det_by_minor.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<det_by_minor::Input, det_by_minor::GradientOutput> {
  CppAD::ADFun<double> *_tape;
public:
  Gradient(det_by_minor::Input& input) : Function(input) {
    std::vector<ADdouble> X(_input.A.size());
    std::copy(_input.A.begin(), _input.A.end(), X.data());
    CppAD::Independent(X);
    std::vector<ADdouble> Y(1);
    det_by_minor::primal<ADdouble>(_input.ell, X.data(), &Y[0]);
    _tape = new CppAD::ADFun<double>(X, Y);
    _tape->optimize("no_compare_op no_conditional_skip no_print_for_op");
  }

  void compute(det_by_minor::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);
    output = _tape->Jacobian(_input.A);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det_by_minor::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

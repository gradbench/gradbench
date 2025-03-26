#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  CppAD::ADFun<double> *_tape;
public:
  Gradient(llsq::Input& input) : Function(input) {
    std::vector<ADdouble> X(_input.x.size());
    std::copy(_input.x.begin(), _input.x.end(), X.data());
    CppAD::Independent(X);
    std::vector<ADdouble> Y(1);
    llsq::primal<ADdouble>(_input.n, _input.x.size(), X.data(), &Y[0]);
    _tape = new CppAD::ADFun<double>(X, Y);
    _tape->optimize("no_compare_op no_conditional_skip no_print_for_op");
  }

  void compute(llsq::GradientOutput& output) {
    output.resize(_input.x.size());
    output = _tape->Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

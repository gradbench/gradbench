#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/an_ode.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<an_ode::Input, an_ode::GradientOutput> {
  CppAD::ADFun<double> *_tape;
public:
  Gradient(an_ode::Input& input) : Function(input) {
    std::vector<ADdouble> X(_input.x.size());
    std::copy(_input.x.begin(), _input.x.end(), X.data());
    CppAD::Independent(X);
    std::vector<ADdouble> Y(1);
    std::vector<ADdouble> Z(_input.x.size());
    an_ode::primal<ADdouble>(_input.x.size(), X.data(), _input.s, Z.data());
    Y[0] = Z.back();
    _tape = new CppAD::ADFun<double>(X, Y);
    _tape->optimize("no_compare_op no_conditional_skip no_print_for_op");
  }

  void compute(an_ode::GradientOutput& output) {
    output.resize(_input.x.size());
    output = _tape->Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<an_ode::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

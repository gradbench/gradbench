#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/ode.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  std::vector<ADdouble> _X, _Y, _Z;

public:
  Gradient(ode::Input& input) :
    Function(input),
    _X(_input.x.size()),
    _Y(1),
    _Z(_input.x.size()) {
    std::copy(_input.x.begin(), _input.x.end(), _X.data());
  }

  void compute(ode::GradientOutput& output) {
    output.resize(_input.x.size());

    CppAD::Independent(_X);
    ode::primal<ADdouble>(_input.x.size(), _X.data(), _input.s, _Z.data());
    _Y[0] = _Z.back();
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<ode::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

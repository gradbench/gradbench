#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"
#include <algorithm>
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<lse::Input, lse::GradientOutput> {
  std::vector<ADdouble> _X, _Y;

public:
  Gradient(lse::Input& input) : Function(input), _X(_input.x.size()), _Y(1) {
    std::copy(_input.x.begin(), _input.x.end(), _X.data());
  }

  void compute(lse::GradientOutput& output) {
    output.resize(_input.x.size());
    CppAD::Independent(_X);
    lse::primal<ADdouble>(_input.x.size(), _X.data(), &_Y[0]);
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<lse::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/logsumexp.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<logsumexp::Input, logsumexp::GradientOutput> {
  std::vector<ADdouble> _X, _Y;
public:
  Gradient(logsumexp::Input& input)
    : Function(input),
      _X(_input.x.size()),
      _Y(1) {
    std::copy(_input.x.begin(), _input.x.end(), _X.data());
  }

  void compute(logsumexp::GradientOutput& output) {
    output.resize(_input.x.size());
    CppAD::Independent(_X);
    logsumexp::primal<ADdouble>(_input.x.size(), _X.data(), &_Y[0]);
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<logsumexp::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

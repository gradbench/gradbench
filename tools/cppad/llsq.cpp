#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  std::vector<ADdouble> _X, _Y;
public:
  Gradient(llsq::Input& input)
    : Function(input),
      _X(_input.x.size()),
      _Y(1) {
    std::copy(_input.x.begin(), _input.x.end(), _X.data());
  }

  void compute(llsq::GradientOutput& output) {
    output.resize(_input.x.size());
    CppAD::Independent(_X);
    llsq::primal<ADdouble>(_input.n, _input.x.size(), _X.data(), &_Y[0]);
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input.x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

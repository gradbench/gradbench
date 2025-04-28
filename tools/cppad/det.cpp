#include "gradbench/evals/det.hpp"
#include "gradbench/main.hpp"
#include <algorithm>
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Gradient : public Function<det::Input, det::GradientOutput> {
  std::vector<ADdouble> _X, _Y;

public:
  Gradient(det::Input& input) : Function(input), _X(_input.A.size()), _Y(1) {
    std::copy(_input.A.begin(), _input.A.end(), _X.data());
  }

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell * ell);

    CppAD::Independent(_X);
    det::primal<ADdouble>(_input.ell, _X.data(), &_Y[0]);
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input.A);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<det::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

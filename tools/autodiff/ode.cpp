#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

using Eigen::VectorXd;

class Gradient : public Function<ode::Input, ode::GradientOutput> {
private:
  VectorXvar _x;

public:
  Gradient(ode::Input& input) : Function(input), _x(input.x.size()) {
    for (auto i = 0; i < _x.size(); i++) {
      _x[i] = _input.x[i];
    }
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    VectorXvar y(n);
    ode::primal<var>(n, _x.data(), _input.s, y.data());
    VectorXd dydx = gradient(y[n - 1], _x);
    for (size_t i = 0; i < n; i++) {
      output[i] = dydx[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<ode::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
}

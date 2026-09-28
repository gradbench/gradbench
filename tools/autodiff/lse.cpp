#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

using Eigen::VectorXd;

class Gradient : public Function<lse::Input, lse::GradientOutput> {
private:
  VectorXvar _x;

public:
  Gradient(lse::Input& input) : Function(input), _x(input.x.size()) {
    for (auto i = 0; i < _x.size(); i++) {
      _x[i] = _input.x[i];
    }
  }

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    var y;
    lse::primal<var>(n, _x.data(), &y);
    VectorXd dydx = gradient(y, _x);
    for (size_t i = 0; i < n; i++) {
      output[i] = dydx[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<lse::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
}

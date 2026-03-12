#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    using Eigen::VectorXd;
    VectorXvar x(m);
    for (size_t i = 0; i < m; i++) {
      x[i] = _input.x[i];
    }
    var y;
    llsq::primal<var>(n, m, x.data(), &y);
    VectorXd dydx = gradient(y, x);
    for (size_t i = 0; i < m; i++) {
      output[i] = dydx[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<llsq::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
}

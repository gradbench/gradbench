#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

using Eigen::VectorXd;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
private:
  VectorXvar _x;

public:
  Gradient(llsq::Input& input) : Function(input), _x(input.x.size()) {
    for (auto i = 0; i < _x.size(); i++) {
      _x[i] = _input.x[i];
    }
  }

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    var y;
    llsq::primal<var>(n, m, _x.data(), &y);
    VectorXd dydx = gradient(y, _x);
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

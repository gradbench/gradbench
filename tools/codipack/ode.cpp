#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"
#include <algorithm>
#include <codi.hpp>

using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  std::vector<Real> _x_d;

public:
  Gradient(ode::Input& input) : Function(input), _x_d(_input.x.size()) {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    Tape& tape = Real::getTape();
    tape.reset();
    tape.setActive();

    for (auto& x : _x_d) {
      tape.registerInput(x);
    }

    std::vector<Real> primal_out(n);
    ode::primal(n, _x_d.data(), _input.s, primal_out.data());

    for (auto& x : primal_out) {
      tape.registerOutput(x);
    }
    tape.setPassive();
    primal_out[n - 1].setGradient(1.0);
    tape.evaluate();

    for (size_t i = 0; i < n; i++) {
      output[i] = _x_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<ode::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

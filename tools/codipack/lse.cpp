#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/lse.hpp"

#include <codi.hpp>

using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Gradient : public Function<lse::Input, lse::GradientOutput> {
  std::vector<Real> _x_d;
public:
  Gradient(lse::Input& input)
    : Function(input),
      _x_d(_input.x.size()){
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    Tape& tape = Real::getTape();
    tape.reset();
    tape.setActive();

    for (auto &x : _x_d) {
      tape.registerInput(x);
    }

    Real error;
    lse::primal(n, _x_d.data(), &error);

    tape.registerOutput(error);
    tape.setPassive();
    error.setGradient(1.0);
    tape.evaluate();

    for (size_t i = 0; i < n; i++) {
      output[i] = _x_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<lse::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

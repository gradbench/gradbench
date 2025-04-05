#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include <codi.hpp>

using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  std::vector<Real> _x_d;
public:
  Gradient(llsq::Input& input) :
    Function(input),
    _x_d(_input.x.size()) {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();

    output.resize(m);

    Tape& tape = Real::getTape();
    tape.reset();
    tape.setActive();

    for (auto &x : _x_d) {
      tape.registerInput(x);
    }

    Real error;
    llsq::primal(n, m, _x_d.data(), &error);

    tape.registerOutput(error);
    tape.setPassive();
    error.setGradient(1.0);
    tape.evaluate();

    for (size_t i = 0; i < m; i++) {
      output[i] = _x_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

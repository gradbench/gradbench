#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det_by_minor.hpp"
#include <codi.hpp>

using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Gradient : public Function<det_by_minor::Input, det_by_minor::GradientOutput> {
  std::vector<Real> _A_d;
public:
  Gradient(det_by_minor::Input& input) :
    Function(input),
    _A_d(_input.A.size()) {
    std::copy(_input.A.begin(), _input.A.end(), _A_d.begin());
  }

  void compute(det_by_minor::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    Tape& tape = Real::getTape();
    tape.reset();
    tape.setActive();

    for (auto &x : _A_d) {
      tape.registerInput(x);
    }

    Real error;
    det_by_minor::primal(ell, _A_d.data(), &error);

    tape.registerOutput(error);
    tape.setPassive();
    error.setGradient(1.0);
    tape.evaluate();

    for (size_t i = 0; i < ell*ell; i++) {
      output[i] = _A_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det_by_minor::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

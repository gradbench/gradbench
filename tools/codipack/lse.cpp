#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/lse.hpp"

#include "codi_impl.hpp"

class Gradient : public Function<lse::Input, lse::GradientOutput>, CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> _x_d;

  Real error;
public:
  Gradient(lse::Input& input)
    : Function(input),
      _x_d(_input.x.size()),
      error() {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    codiStartRecording();

    for (auto &x : _x_d) {
      codiAddInput(x);
    }

    lse::primal(n, _x_d.data(), &error);

    codiAddOutput(error);
    codiStopRecording();

    codiSetGradient(error, 1.0);
    codiEval();

    for (size_t i = 0; i < n; i++) {
      output[i] = codiGetGradient(_x_d[i]);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<lse::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

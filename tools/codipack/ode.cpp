#include "gradbench/evals/ode.hpp"
#include "codi_impl.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<ode::Input, ode::GradientOutput>,
                 CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> _x_d;
  std::vector<Real> primal_out;

public:
  Gradient(ode::Input& input)
      : Function(input), _x_d(_input.x.size()), primal_out(input.x.size()) {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    codiStartRecording();

    for (auto& x : _x_d) {
      codiAddInput(x);
    }

    ode::primal(n, _x_d.data(), _input.s, primal_out.data());

    for (auto& x : primal_out) {
      codiAddOutput(x);
    }
    codiStopRecording();

    codiSetGradient(primal_out[n - 1], 1.0);
    codiEval();

    for (size_t i = 0; i < n; i++) {
      output[i] = codiGetGradient(_x_d[i]);
    }

    codiCleanup();
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

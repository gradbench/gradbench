#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;
  // Inputs live across calls, as in the other C++ tools; clearAll resets the
  // tape, so registerInputs gives them fresh slots on every call.
  std::vector<AD> x;
  // Reuses its memory; cleared before the call ends, while its slots are valid.
  std::vector<AD> y;

public:
  Gradient(ode::Input& input)
      : Function(input), x(_input.x.begin(), _input.x.end()) {}

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _tape.clearAll();
    _tape.registerInputs(x);
    _tape.newRecording();

    y.resize(n);
    ode::primal(n, x.data(), _input.s, y.data());

    _tape.registerOutputs(y);
    xad::derivative(y[n - 1]) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < n; i++) {
      output[i] = xad::derivative(x[i]);
    }
    y.clear();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<ode::Primal>},
                       {"gradient", function_main<Gradient>}});
}

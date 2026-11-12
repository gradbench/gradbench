#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Gradient : public Function<lse::Input, lse::GradientOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;
  // Inputs live across calls, as in the other C++ tools; clearAll resets the
  // tape, so registerInputs gives them fresh slots on every call.
  std::vector<AD> x;

public:
  Gradient(lse::Input& input)
      : Function(input), x(_input.x.begin(), _input.x.end()) {}

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _tape.clearAll();
    _tape.registerInputs(x);
    _tape.newRecording();

    AD y;
    lse::primal(n, x.data(), &y);

    _tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < n; i++) {
      output[i] = xad::derivative(x[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<lse::Primal>},
                       {"gradient", function_main<Gradient>}});
}

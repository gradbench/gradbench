#include "gradbench/evals/det.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Gradient : public Function<det::Input, det::GradientOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;

public:
  Gradient(det::Input& input) : Function(input) {}

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell * ell);

    _tape.clearAll();
    std::vector<AD> A(_input.A.begin(), _input.A.end());
    _tape.registerInputs(A);
    _tape.newRecording();

    AD y;
    det::primal(ell, A.data(), &y);

    _tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < ell * ell; i++) {
      output[i] = xad::derivative(A[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<det::Primal>},
                       {"gradient", function_main<Gradient>}});
}

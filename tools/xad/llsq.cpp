#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;

public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    _tape.clearAll();
    std::vector<AD> x(_input.x.begin(), _input.x.end());
    _tape.registerInputs(x);
    _tape.newRecording();

    AD y;
    llsq::primal(n, m, x.data(), &y);

    _tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < m; i++) {
      output[i] = xad::derivative(x[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<llsq::Primal>},
                       {"gradient", function_main<Gradient>}});
}

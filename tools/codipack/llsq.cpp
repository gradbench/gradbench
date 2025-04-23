#include "gradbench/evals/llsq.hpp"
#include "codi_impl.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<llsq::Input, llsq::GradientOutput>,
                 CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> _x_d;
  Real              error;

public:
  Gradient(llsq::Input& input)
      : Function(input), _x_d(_input.x.size()), error() {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
  }

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();

    output.resize(m);

    codiStartRecording();

    for (auto& x : _x_d) {
      codiAddInput(x);
    }

    llsq::primal(n, m, _x_d.data(), &error);

    codiAddOutput(error);
    codiStopRecording();

    codiSetGradient(error, 1.0);
    codiEval();

    for (size_t i = 0; i < m; i++) {
      output[i] = codiGetGradient(_x_d[i]);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<llsq::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

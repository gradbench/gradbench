#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det.hpp"

#include "codi_impl.hpp"


class Gradient : public Function<det::Input, det::GradientOutput>, CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> _A_d;

  Real error;
public:
  Gradient(det::Input& input) :
    Function(input),
    _A_d(_input.A.size()),
    error()
  {
    std::copy(_input.A.begin(), _input.A.end(), _A_d.begin());
  }

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    codiStartRecording();

    for (auto &x : _A_d) {
      codiAddInput(x);
    }

    det::primal(ell, _A_d.data(), &error);

    codiAddOutput(error);
    codiStopRecording();

    codiSetGradient(error, 1.0);
    codiEval();

    for (size_t i = 0; i < ell*ell; i++) {
      output[i] = codiGetGradient(_A_d[i]);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det::Primal>},
      {"gradient", function_main<Gradient>},
    });
}

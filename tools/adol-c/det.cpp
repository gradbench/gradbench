#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/det.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Gradient : public Function<det::Input, det::GradientOutput> {
public:
  Gradient(det::Input& input) : Function(input) {
    size_t ell = _input.ell;

    trace_on(tapeTag);

    std::vector<adouble> A_d(ell*ell);
    for (size_t i = 0; i < ell*ell; i++) {
      A_d[i] <<= _input.A[i];
    }

    adouble primal_out_d;
    det::primal(ell, A_d.data(), &primal_out_d);

    double primal_out;
    primal_out_d >>= primal_out;

    trace_off();
  }

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    gradient(tapeTag, ell*ell, _input.A.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

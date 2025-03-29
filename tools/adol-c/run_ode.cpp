#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/ode.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Gradient : public Function<ode::Input, ode::GradientOutput> {
public:
  Gradient(ode::Input& input) : Function(input) {
    size_t n = _input.x.size();

    trace_on(tapeTag);

    std::vector<adouble> x_d(n);
    for (size_t i = 0; i < n; i++) {
      x_d[i] <<= _input.x[i];
    }

    std::vector<adouble> primal_out_d(n);
    ode::primal(n, x_d.data(), _input.s, primal_out_d.data());

    double primal_out;
    primal_out_d.back() >>= primal_out;

    trace_off();
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    gradient(tapeTag, n, _input.x.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<ode::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

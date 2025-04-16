#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <algorithm>
#include <vector>

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {
    size_t n = _input.n;
    size_t m = _input.x.size();

    trace_on(tapeTag);

    std::vector<adouble> x_d(m);
    for (size_t i = 0; i < m; i++) {
      x_d[i] <<= _input.x[i];
    }

    adouble primal_out_d;
    llsq::primal(n, m, x_d.data(), &primal_out_d);

    double primal_out;
    primal_out_d >>= primal_out;

    trace_off();
  }

  void compute(llsq::GradientOutput& output) {
    size_t m = _input.x.size();
    output.resize(m);

    gradient(tapeTag, m, _input.x.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<llsq::Primal>},
                       {"gradient", function_main<Gradient>}});
}

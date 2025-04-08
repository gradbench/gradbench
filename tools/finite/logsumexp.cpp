#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/logsumexp.hpp"
#include "finite.h"

class Gradient : public Function<logsumexp::Input, logsumexp::GradientOutput> {
private:
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(logsumexp::Input& input) : Function(input), _engine(_input.x.size()) {}

  void compute(logsumexp::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _engine.finite_differences(1, [&](double* in, double* out) {
      logsumexp::primal<double>(n, in, out);
    }, _input.x.data(), n, 1, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<logsumexp::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

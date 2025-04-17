#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<lse::Input, lse::GradientOutput> {
public:
  Gradient(lse::Input& input) : Function(input) {}

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    double max_elem = *std::max_element(_input.x.begin(), _input.x.end());

    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
      output[i] = std::exp(_input.x[i] - max_elem);
      sum += output[i];
    }

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      output[i] /= sum;
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<lse::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

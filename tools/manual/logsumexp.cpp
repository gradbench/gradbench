#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/logsumexp.hpp"

class Gradient : public Function<logsumexp::Input, logsumexp::GradientOutput> {
public:
  Gradient(logsumexp::Input& input) : Function(input) {}

  void compute(logsumexp::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    double max_elem = *std::max_element(_input.x.begin(), _input.x.end());

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      output[i] = std::exp(_input.x[i] - max_elem);
      sum += output[i];
    }

    for (size_t i = 0; i < n; ++i) {
      output[i] /= sum;
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<logsumexp::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

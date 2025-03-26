#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    std::vector<double> sums(n);
    for (size_t i = 0; i < n; i++) {
      double ti = llsq::t(i, n);
      double inner_sum = llsq::s(ti);
      for (size_t l = 0; l < m; l++) {
        inner_sum -= _input.x[l] * pow(ti, l);
      }
      sums[i] = inner_sum;
    }
    for(size_t j = 0; j < m; j++) {
      double sum = 0;
      for (size_t i = 0; i < n; i++) {
        double ti = llsq::t(i, n);
        sum += sums[i]*pow(ti,j);
      }
      output[j] = -sum;
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/an_ode.hpp"

class Gradient : public Function<an_ode::Input, an_ode::GradientOutput> {
public:
  Gradient(an_ode::Input& input) : Function(input) {}

  void compute(an_ode::GradientOutput& output) {
    output.resize(_input.x.size());

    size_t n = _input.x.size();
    size_t r = n-1;

    double tf  = 2.0;
    double y_r = _input.x[0] * tf;
    for(size_t i = 1; i <= r; i++) {
      y_r = y_r * _input.x[i] * tf / double(i+1);
    }

    for (size_t i = 0; i < n; i++) {
      output[i] = y_r / _input.x[i];
    }
  }
};


int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<an_ode::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

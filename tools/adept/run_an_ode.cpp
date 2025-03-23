#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/an_ode.hpp"
#include "adept.h"

using adept::adouble;

class Gradient : public Function<an_ode::Input, an_ode::GradientOutput> {
public:
  Gradient(an_ode::Input& input) : Function(input) {}

  void compute(an_ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adept::Stack stack;
    std::vector<adouble> x_d(n);

    adept::set_values(x_d.data(), n, _input.x.data());

    stack.new_recording();
    std::vector<adouble> primal_out(n);
    an_ode::primal(n, x_d.data(), _input.s, primal_out.data());
    primal_out.back().set_gradient(1.);
    stack.reverse();

    adept::get_gradients(x_d.data(), n, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<an_ode::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

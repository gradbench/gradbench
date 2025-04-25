#include "gradbench/evals/ode.hpp"
#include "adept.h"
#include "gradbench/main.hpp"
#include <algorithm>
#include <vector>

using adept::adouble;

class Gradient : public Function<ode::Input, ode::GradientOutput> {
public:
  Gradient(ode::Input& input) : Function(input) {}

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adept::Stack         stack;
    std::vector<adouble> x_d(n);

    adept::set_values(x_d.data(), n, _input.x.data());

    stack.new_recording();
    std::vector<adouble> primal_out(n);
    ode::primal(n, x_d.data(), _input.s, primal_out.data());
    primal_out.back().set_gradient(1.);
    stack.reverse();

    adept::get_gradients(x_d.data(), n, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<ode::Primal>},
                       {"gradient", function_main<Gradient>}});
}

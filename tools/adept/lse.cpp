#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lse.hpp"
#include "adept.h"

using adept::adouble;

class Gradient : public Function<lse::Input, lse::GradientOutput> {
public:
  Gradient(lse::Input& input) : Function(input) {}

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adept::Stack stack;
    std::vector<adouble> x_d(n);
    adept::set_values(x_d.data(), n, _input.x.data());

    stack.new_recording();
    adouble primal_out;
    lse::primal(n, x_d.data(), &primal_out);
    primal_out.set_gradient(1.);
    stack.reverse();

    adept::get_gradients(x_d.data(), n, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<lse::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include "adept.h"

using adept::adouble;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    adept::Stack stack;
    std::vector<adouble> x_d(m);

    adept::set_values(x_d.data(), m, _input.x.data());

    stack.new_recording();
    adouble primal_out;
    llsq::primal(n, m, x_d.data(), &primal_out);
    primal_out.set_gradient(1.);
    stack.reverse();

    adept::get_gradients(x_d.data(), m, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

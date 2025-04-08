#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/logsumexp.hpp"
#include "adept.h"

using adept::adouble;

class Gradient : public Function<logsumexp::Input, logsumexp::GradientOutput> {
public:
  Gradient(logsumexp::Input& input) : Function(input) {}

  void compute(logsumexp::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adept::Stack stack;
    std::vector<adouble> x_d(n);
    adept::set_values(x_d.data(), n, _input.x.data());

    stack.new_recording();
    adouble primal_out;
    logsumexp::primal(n, x_d.data(), &primal_out);
    primal_out.set_gradient(1.);
    stack.reverse();

    adept::get_gradients(x_d.data(), n, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<logsumexp::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

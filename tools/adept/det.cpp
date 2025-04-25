#include "gradbench/evals/det.hpp"
#include "adept.h"
#include "gradbench/main.hpp"
#include <algorithm>
#include <vector>

using adept::adouble;

class Gradient : public Function<det::Input, det::GradientOutput> {
public:
  Gradient(det::Input& input) : Function(input) {}

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell * ell);

    adept::Stack         stack;
    std::vector<adouble> A_d(ell * ell);

    adept::set_values(A_d.data(), ell * ell, _input.A.data());

    stack.new_recording();
    adouble primal_out;
    det::primal(ell, A_d.data(), &primal_out);
    primal_out.set_gradient(1.);
    stack.reverse();

    adept::get_gradients(A_d.data(), ell * ell, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<det::Primal>},
                       {"gradient", function_main<Gradient>}});
}

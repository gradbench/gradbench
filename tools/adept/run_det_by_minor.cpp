#include <algorithm>
#include <vector>
#include "gradbench/main.hpp"
#include "gradbench/evals/det_by_minor.hpp"
#include "adept.h"

using adept::adouble;

class Gradient : public Function<det_by_minor::Input, det_by_minor::GradientOutput> {
public:
  Gradient(det_by_minor::Input& input) : Function(input) {}

  void compute(det_by_minor::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    adept::Stack stack;
    std::vector<adouble> A_d(ell*ell);

    adept::set_values(A_d.data(), ell*ell, _input.A.data());

    stack.new_recording();
    adouble primal_out;
    det_by_minor::primal(ell, A_d.data(), &primal_out);
    primal_out.set_gradient(1.);
    stack.reverse();

    adept::get_gradients(A_d.data(), ell*ell, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det_by_minor::Primal>},
      {"gradient", function_main<Gradient>}
    });
}

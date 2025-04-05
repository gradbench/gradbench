#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det.hpp"

class Gradient : public Function<det::Input, det::GradientOutput> {
public:
  Gradient(det::Input& input) : Function(input) {}

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    std::vector<size_t> r(ell + 1);
    std::vector<size_t> c(ell + 1);
    for(size_t i = 0; i < ell; i++) {
      r[i] = i+1;
      c[i] = i+1;
    }

    for (size_t i = 0; i < ell; i++) {
      for (size_t j = 0; j < ell; j++) {
        r[i==0?ell:i-1] = i+1;
        c[j==0?ell:j-1] = j+1;
        double M = det::det_of_minor(_input.A.data(), ell, ell-1, r, c);
        output[i*ell+j] = pow(-1, i+j) * M;
        r[i==0?ell:i-1] = i;
        c[j==0?ell:j-1] = j;
      }
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}

#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#pragma omp declare reduction(                                                 \
        vec_double_plus : std::vector<double> : std::transform(                \
                omp_out.begin(), omp_out.end(), omp_in.begin(),                \
                    omp_out.begin(), std::plus<double>()))                     \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    std::vector<double> sums(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      double ti        = llsq::t(i, n);
      double inner_sum = llsq::s(ti);
      double mul       = 1;
      for (size_t l = 0; l < m; l++) {
        inner_sum -= _input.x[l] * mul;
        mul *= ti;
      }
      sums[i] = inner_sum;
    }

    for (size_t j = 0; j < m; j++) {
      output[j] = 0;
    }

#pragma omp parallel for reduction(vec_double_plus : output)
    for (size_t i = 0; i < n; i++) {
      double ti   = llsq::t(i, n);
      double term = 1.0;

      for (size_t j = 0; j < m; j++) {
        output[j] -= sums[i] * term;
        term *= ti;
      }
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<llsq::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}

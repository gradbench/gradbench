#include "gradbench/evals/saddle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"
#include "solver.hpp"

// A wrapper with an 'out'-parameter as I find it easier to reason
// about that when using Enzyme.
void primal(const double* v, double* out) {
  *out = saddle::objective(v[0], v[1], v[2], v[3]);
}

class RR : public Function<saddle::Input, saddle::Output> {
  struct Inner {

    void objective(const double* v, double* out) const {
      primal(v, out);
    }

    void gradient(const double* v, double* out) const {
      double dummy, unit = 1;
      __enzyme_autodiff(primal,
                        enzyme_dup, v, out,
                        enzyme_dupnoneed, &dummy, &unit);
    }

    size_t input_size() const { return 4; }
  };

  struct Outer {
    std::vector<double> start;

    Outer(const std::vector<double>& v) : start(v) {}

    static void objective_static(size_t n,
                                 const double* start,
                                 const double* v,
                                 double* out) {
      *out = multivariate_max(Inner(), start);
    }

    void objective(const double* v, double* out) const {
      objective_static(2, start.data(), v, out);
    }

    void gradient(const double* v, double* out) const {
      double dummy, unit = 1;
      __enzyme_autodiff(objective_static,
                        enzyme_const, (size_t)2,
                        enzyme_const, start.data(),
                        enzyme_dup, v, out,
                        enzyme_dupnoneed, &dummy, &unit);
    }

    size_t input_size() const { return 2; }
  };

public:
  RR(saddle::Input& input) : Function(input) {}
  void compute(saddle::Output& output) {
    multivariate_argmin<Outer>(Outer(_input), _input.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<RR>}
    });
}

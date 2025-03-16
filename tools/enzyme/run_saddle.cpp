#include "gradbench/evals/saddle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"
#include "solver.hpp"

// A wrapper with an 'out'-parameter as I find it easier to reason
// about that when using Enzyme.
void primal(const double* v, double* out) {
  *out = saddle::objective(v[0], v[1], v[2], v[3]);
}

struct MaxPrimalR {
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

void max_primal_r(size_t n,
                  const double* start,
                  const double* v,
                  double* out) {

  *out = multivariate_max(MaxPrimalR(), start);
}

class RR : public Function<saddle::Input, saddle::Output> {
  struct Outer {
    double _start[2];

    Outer(const saddle::Input& input) {
      _start[0] = input.start[0];
      _start[1] = input.start[1];
    }

    void objective(const double* v, double* out) const {
      max_primal_r(2, _start, v, out);
    }

    void gradient(const double* v, double* out) const {
      double dummy, unit = 1;
      __enzyme_autodiff(max_primal_r,
                        enzyme_const, (size_t)2,
                        enzyme_const, _start,
                        enzyme_dup, v, out,
                        enzyme_dupnoneed, &dummy, &unit);
    }

    size_t input_size() const { return 2; }
  };

  std::vector<double> _start;

public:
  RR(saddle::Input& input) : Function(input), _start(2) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }
  void compute(saddle::Output& output) {
    output = multivariate_argmin<Outer>(Outer(_input), _input.start);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<RR>}
    });
}

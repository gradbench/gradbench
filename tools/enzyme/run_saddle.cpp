#include "gradbench/evals/saddle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"
#include "solver.hpp"

struct double2{ double x, y; };

struct R2Cost {
  const double *_p1;

  R2Cost(const double *p1) : _p1(p1) {}

  void objective(const double* p2, double* out) const {
    *out = saddle::objective(_p1[0], _p1[1], p2[0], p2[1]);
  }

  void gradient(const double* p2, double* out) const {
    auto [p2x, p2y] = __enzyme_autodiff_template<double2>
      ((void*)saddle::objective<double>,
       enzyme_const, _p1[0],
       enzyme_const, _p1[1],
       enzyme_out, p2[0],
       enzyme_out, p2[1]);
    out[0] = p2x;
    out[1] = p2y;
  }

  size_t input_size() const { return 2; }
};

void max_primal_r(const double *p1,
                  const double *p2,
                  double* out) {
  *out = multivariate_max(R2Cost(p1), p2);
}

struct R1Cost {
  double _start[2];

  R1Cost(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* v, double* out) const {
    max_primal_r(_start, v, out);
  }

  void gradient(const double* v, double* out) const {
    double dummy, unit = 1;
    out[0] = 0;
    out[1] = 0;
    __enzyme_autodiff(max_primal_r,
                      enzyme_const, _start,
                      enzyme_dup, v, out,
                      enzyme_dupnoneed, &dummy, &unit);
  }

  size_t input_size() const { return 2; }
};

class RR : public Function<saddle::Input, saddle::Output> {
  std::vector<double> _start;

public:
  RR(saddle::Input& input) : Function(input), _start(2) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }
  void compute(saddle::Output& output) {
    auto r1 = multivariate_argmin(R1Cost(_input), _input.start);
    auto r2 = std::vector<double>{0,0};
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[2];
    output[3] = r2[3];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<RR>}
    });
}

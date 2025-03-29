// This implements all variants with finite differences, so it is not
// a faithful implementation. It is provided solely as a performance
// comparison. Because the domain is so small, finite differences can
// be expected to perform fairly well on this problem, which is not
// the case for most benchmarks.

#include "gradbench/evals/saddle.hpp"
#include "gradbench/main.hpp"
#include "gradbench/gd.hpp"
#include "finite.h"

struct R2Cost {
  const double *_p1;

  size_t input_size() const { return 2; }

  R2Cost(const double *p1) : _p1(p1) {}

  void objective(const double* p2, double* out) const {
    *out = saddle::objective(_p1[0], _p1[1], p2[0], p2[1]);
  }

  void gradient(const double* p2, double* out) const {
    FiniteDifferencesEngine<double> engine(input_size());
    double p2_copy[2] = {p2[0], p2[1]};
    engine.finite_differences(1, [&](double *fd_in, double* fd_out) {
      objective(fd_in, fd_out);
    }, p2_copy, input_size(), 1, out);
  }
};

class R1Cost {
  double _start[2];
public:
  size_t input_size() const { return 2; }

  R1Cost(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* p1, double* out) const {
    *out = multivariate_max(R2Cost(p1), _start);
  }

  void gradient(const double* p1, double* out) const {
    FiniteDifferencesEngine<double> engine(input_size());
    double p1_copy[2] = {p1[0], p1[1]};
    engine.finite_differences(1, [&](double *fd_in, double* fd_out) {
      objective(fd_in, fd_out);
    }, p1_copy, input_size(), 1, out);
  }
};

class Saddle : public Function<saddle::Input, saddle::Output> {
public:
  Saddle(saddle::Input& input) : Function(input) {}
  void compute(saddle::Output& output) {
    auto r1 = multivariate_argmin(R1Cost(_input), _input.start);
    auto r2 = multivariate_argmax(R2Cost(r1.data()), _input.start);
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[0];
    output[3] = r2[1];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<Saddle>},
      {"rf", function_main<Saddle>},
      {"ff", function_main<Saddle>},
      {"fr", function_main<Saddle>}
    });
}

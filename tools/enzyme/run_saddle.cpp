#include "gradbench/evals/saddle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"
#include "solver.hpp"

struct double2{ double x, y; };

struct R2CostR {
  const double *_p1;

  R2CostR(const double *p1) : _p1(p1) {}

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

double max_primal_r(double p1x, double p1y,
                    double p2x, double p2y) {
  double p1[2] = {p1x, p1y};
  double p2[2] = {p2x, p2y};
  return multivariate_max(R2CostR(p1), p2);
}

struct R2CostF {
  const double *_p1;

  R2CostF(const double *p1) : _p1(p1) {}

  void objective(const double* p2, double* out) const {
    *out = saddle::objective(_p1[0], _p1[1], p2[0], p2[1]);
  }

  void gradient(const double* p2, double* out) const {
    out[0] = __enzyme_fwddiff_template<double>
      ((void*)saddle::objective<double>,
       enzyme_const, _p1[0],
       enzyme_const, _p1[1],
       enzyme_dup, p2[0], 1.0,
       enzyme_dup, p2[1], 0.0);
    out[1] = __enzyme_fwddiff_template<double>
      ((void*)saddle::objective<double>,
       enzyme_const, _p1[0],
       enzyme_const, _p1[1],
       enzyme_dup, p2[0], 0.0,
       enzyme_dup, p2[1], 1.0);
  }

  size_t input_size() const { return 2; }
};

double max_primal_f(double p1x, double p1y,
                    double p2x, double p2y) {
  double p1[2] = {p1x, p1y};
  double p2[2] = {p2x, p2y};
  return multivariate_max(R2CostF(p1), p2);
}

struct R1CostRR {
  double _start[2];

  R1CostRR(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* p1, double* out) const {
    *out = max_primal_r(p1[0], p1[1], _start[0], _start[1]);
  }

  void gradient(const double* p1, double* out) const {
    auto [p1x, p1y] =
      __enzyme_autodiff_template<double2>((void*)max_primal_r,
                                          enzyme_out, p1[0],
                                          enzyme_out, p1[1],
                                          enzyme_const, _start[0],
                                          enzyme_const, _start[1]);
    out[0] = p1x;
    out[1] = p1y;
  }

  size_t input_size() const { return 2; }
};

struct R1CostRF {
  double _start[2];

  R1CostRF(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* p1, double* out) const {
    *out = max_primal_f(p1[0], p1[1], _start[0], _start[1]);
  }

  void gradient(const double* p1, double* out) const {
    auto [p1x, p1y] =
      __enzyme_autodiff_template<double2>((void*)max_primal_f,
                                          enzyme_out, p1[0],
                                          enzyme_out, p1[1],
                                          enzyme_const, _start[0],
                                          enzyme_const, _start[1]);
    out[0] = p1x;
    out[1] = p1y;
  }

  size_t input_size() const { return 2; }
};

struct R1CostFF {
  double _start[2];

  R1CostFF(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* p1, double* out) const {
    *out = max_primal_f(p1[0], p1[1], _start[0], _start[1]);
  }

  void gradient(const double* p1, double* out) const {
    out[0] = __enzyme_fwddiff_template<double>
      ((void*)max_primal_f,
       enzyme_dup, p1[0], 1.0,
       enzyme_dup, p1[1], 0.0,
       enzyme_const, _start[0],
       enzyme_const, _start[1]);
    out[1] = __enzyme_fwddiff_template<double>
      ((void*)max_primal_f,
       enzyme_dup, p1[0], 0.0,
       enzyme_dup, p1[1], 1.0,
       enzyme_const, _start[0],
       enzyme_const, _start[1]);
  }

  size_t input_size() const { return 2; }
};

struct R1CostFR {
  double _start[2];

  R1CostFR(const saddle::Input& input) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }

  void objective(const double* p1, double* out) const {
    *out = max_primal_r(p1[0], p1[1], _start[0], _start[1]);
  }

  void gradient(const double* p1, double* out) const {
    out[0] = __enzyme_fwddiff_template<double>
      ((void*)max_primal_r,
       enzyme_dup, p1[0], 1.0,
       enzyme_dup, p1[1], 0.0,
       enzyme_const, _start[0],
       enzyme_const, _start[1]);
    out[1] = __enzyme_fwddiff_template<double>
      ((void*)max_primal_r,
       enzyme_dup, p1[0], 0.0,
       enzyme_dup, p1[1], 1.0,
       enzyme_const, _start[0],
       enzyme_const, _start[1]);
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
    auto r1 = multivariate_argmin(R1CostRR(_input), _input.start);
    auto r2 = multivariate_argmax(R2CostR(r1.data()), _input.start);
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[0];
    output[3] = r2[1];
  }
};

class FF : public Function<saddle::Input, saddle::Output> {
  std::vector<double> _start;

public:
  FF(saddle::Input& input) : Function(input), _start(2) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }
  void compute(saddle::Output& output) {
    auto r1 = multivariate_argmin(R1CostFF(_input), _input.start);
    auto r2 = multivariate_argmax(R2CostF(r1.data()), _input.start);
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[0];
    output[3] = r2[1];
  }
};

class FR : public Function<saddle::Input, saddle::Output> {
  std::vector<double> _start;

public:
  FR(saddle::Input& input) : Function(input), _start(2) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }
  void compute(saddle::Output& output) {
    auto r1 = multivariate_argmin(R1CostFR(_input), _input.start);
    auto r2 = multivariate_argmax(R2CostR(r1.data()), _input.start);
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[0];
    output[3] = r2[1];
  }
};

class RF : public Function<saddle::Input, saddle::Output> {
  std::vector<double> _start;

public:
  RF(saddle::Input& input) : Function(input), _start(2) {
    _start[0] = input.start[0];
    _start[1] = input.start[1];
  }
  void compute(saddle::Output& output) {
    auto r1 = multivariate_argmin(R1CostRF(_input), _input.start);
    auto r2 = multivariate_argmax(R2CostF(r1.data()), _input.start);
    output.resize(4);
    output[0] = r1[0];
    output[1] = r1[1];
    output[2] = r2[0];
    output[3] = r2[1];
  }
};


int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<RR>},
      {"ff", function_main<FF>},
      {"fr", function_main<FR>},
      {"rf", function_main<RF>}
    });
}

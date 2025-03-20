// This implementation looks more awkward than is normally the case
// for Enzyme, because we want to handle all the four variants (FF,
// FR, RF, RR) without duplicating any core logic. This is largely
// done through templates and an object that encapsulates both the
// 'objective' and 'gradient' forms of a function - but note that
// these gradients are *not* hand-written; they are produced by
// Enzyme.

#include "gradbench/evals/particle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"
#include "solver.hpp"

using particle::Point, particle::accel;

void accel_wrap(const std::vector<Point<double>>* charges,
                const Point<double>* px,
                double* out) {
  *out = accel(*charges, *px);
}

double naive_euler_r(double w) {
  std::vector<Point<double>> charges = { Point(10.0,10.0-w), Point(10.0,0.0) };
  Point<double> x = Point(0.0, 8.0);
  Point<double> xdot = Point(0.75, 0.0);
  double delta_t = 1e-1;

  while (true) {
    Point<double> xddot;
    double dummy, unit = 1;
    __enzyme_autodiff(accel_wrap,
                      enzyme_const, &charges,
                      enzyme_dup, &x, &xddot,
                      enzyme_dupnoneed, &dummy, &unit);
    xddot = -1.0 * xddot;
    x += delta_t * xdot;
    if (x.y <= 0) {
      auto delta_t_f = -x.y / xdot.y;
      auto x_t_f = x + delta_t_f * xdot;
      return x_t_f.x * x_t_f.x;
    }
    xdot += delta_t * xddot;
  }
}

double naive_euler_f(double w) {
  std::vector<Point<double>> charges = { Point(10.0,10.0-w), Point(10.0,0.0) };
  Point<double> x = Point(0.0, 8.0);
  Point<double> xdot = Point(0.75, 0.0);
  double delta_t = 1e-1;

  while (true) {
    Point<double> xddot;
    double dummy = 1;

    {
      Point<double> xdtan(1,0);
    __enzyme_fwddiff(accel_wrap,
                     enzyme_const, &charges,
                     enzyme_dup, &x, &xdtan,
                     enzyme_dupnoneed, &dummy, &xddot.x);
    }
    {
      Point<double> xdtan(0,1);
      __enzyme_fwddiff(accel_wrap,
                       enzyme_const, &charges,
                       enzyme_dup, &x, &xdtan,
                       enzyme_dupnoneed, &dummy, &xddot.y);
    }

    xddot = -1.0 * xddot;
    x += delta_t * xdot;
    if (x.y <= 0) {
      auto delta_t_f = -x.y / xdot.y;
      auto x_t_f = x + delta_t_f * xdot;
      return x_t_f.x * x_t_f.x;
    }
    xdot += delta_t * xddot;
  }
}

class RR : public Function<particle::Input, particle::Output> {
public:
  struct O {
    static void objective(const double* x, double* out) {
      *out = naive_euler_r(x[0]);
    }
    void gradient(const double* x, double* out) const {
      double dummy, unit = 1;
      out[0] = 0;
      __enzyme_autodiff(objective,
                        enzyme_dup, x, out,
                        enzyme_dupnoneed, &dummy, &unit);

    }

    size_t input_size() const { return 1; }
  };

  RR(particle::Input& input) : Function(input) {}
  void compute(particle::Output& output) {
    output = multivariate_argmin(O(), &_input.w0)[0];
  }
};

class RF : public Function<particle::Input, particle::Output> {
public:
  struct O {
    static void objective(const double* x, double* out) {
      *out = naive_euler_f(x[0]);
    }
    void gradient(const double* x, double* out) const {
      double dummy, unit = 1;
      out[0] = 0;
      __enzyme_autodiff(objective,
                        enzyme_dup, x, out,
                        enzyme_dupnoneed, &dummy, &unit);

    }

    size_t input_size() const { return 1; }
  };

  RF(particle::Input& input) : Function(input) {}
  void compute(particle::Output& output) {
    output = multivariate_argmin(O(), &_input.w0)[0];
  }
};

class FR : public Function<particle::Input, particle::Output> {
public:
  struct O {
    static void objective(const double* x, double* out) {
      *out = naive_euler_r(x[0]);
    }
    void gradient(const double* x, double* out) const {
      double dummy, unit = 1;
      out[0] = 0;
      __enzyme_fwddiff(objective,
                       enzyme_dup, x, &unit,
                       enzyme_dupnoneed, &dummy, out);

    }

    size_t input_size() const { return 1; }
  };

  FR(particle::Input& input) : Function(input) {}
  void compute(particle::Output& output) {
    output = multivariate_argmin(O(), &_input.w0)[0];
  }
};

class FF : public Function<particle::Input, particle::Output> {
public:
  struct O {
    static void objective(const double* x, double* out) {
      *out = naive_euler_f(x[0]);
    }
    void gradient(const double* x, double* out) const {
      double dummy, unit = 1;
      out[0] = 0;
      __enzyme_fwddiff(objective,
                       enzyme_dup, x, &unit,
                       enzyme_dupnoneed, &dummy, out);

    }

    size_t input_size() const { return 1; }
  };

  FF(particle::Input& input) : Function(input) {}
  void compute(particle::Output& output) {
    output = multivariate_argmin(O(), &_input.w0)[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<RR>},
      {"rf", function_main<RF>},
      {"ff", function_main<FF>},
      {"fr", function_main<FR>}
    });
}

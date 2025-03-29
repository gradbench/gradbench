// This implements all variants with finite differences, so it is not
// a faithful implementation. It is provided solely as a performance
// comparison. Because the domain is so small, finite differences can
// be expected to perform fairly well on this problem, which is not
// the case for most benchmarks.

#include "gradbench/evals/particle.hpp"
#include "gradbench/main.hpp"
#include "gradbench/gd.hpp"
#include "finite.h"

using particle::Point, particle::accel;

struct Outer {
  Outer() {}
  size_t input_size() const { return 1; }

  static void objective(const double* in, double* out) {
    double w = in[0];
    FiniteDifferencesEngine<double> engine(2);

    std::vector<Point<double>> charges = { Point(10.0,10.0-w), Point(10.0,0.0) };
    Point<double> x = Point(0.0, 8.0);
    Point<double> xdot = Point(0.75, 0.0);
    double delta_t = 1e-1;

    while (true) {
      double x_arr[2] = {x.x, x.y};
      double xddot_arr[2];
      engine.finite_differences(1, [&](double *fd_in, double *res) {
        *res = accel(charges, particle::Point(fd_in[0], fd_in[1]));
      }, x_arr, 2, 1, xddot_arr);
      Point<double> xddot(xddot_arr[0], xddot_arr[1]);

      xddot = -1.0 * xddot;
      x += delta_t * xdot;
      if (x.y <= 0) {
        auto delta_t_f = -x.y / xdot.y;
        auto x_t_f = x + delta_t_f * xdot;
        *out = x_t_f.x * x_t_f.x;
        return;
      }
      xdot += delta_t * xddot;
    }
  }

  void gradient(const double* x, double* out) const {
    FiniteDifferencesEngine<double> engine(input_size());
    double w = x[0];
    engine.finite_differences(1, [&](double *x_per, double *res) {
      objective(x_per, res);
    }, &w, input_size(), 1, out);
  }
};

class Particle : public Function<particle::Input, particle::Output> {
public:
  Particle(particle::Input& input) : Function(input) {}
  void compute(particle::Output& output) {
    output = multivariate_argmin(Outer(), &_input.w0)[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<Particle>},
      {"rf", function_main<Particle>},
      {"ff", function_main<Particle>},
      {"fr", function_main<Particle>}
    });
}

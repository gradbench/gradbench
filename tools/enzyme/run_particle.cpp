#include "gradbench/evals/particle.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"

using particle::Point, particle::accel;

template <typename T>
T magnitude_squared(const std::vector<T>& v) {
  T acc = 0.0;
  for (auto x : v) {
    acc += x*x;
  }
  return acc;
}

template <typename T>
T magnitude(const std::vector<T>& v) {
  return sqrt(magnitude_squared(v));
}

template <typename T>
T vector_dist(const std::vector<T>& u, const std::vector<T>& v) {
  T acc = 0.0;
  for (int i = 0; i < u.size(); i++) {
    acc += (u[i]-v[i])*(u[i]-v[i]);
  }
  return sqrt(acc);
}

std::vector<double> multivariate_argmin(void (*f)(const std::vector<double>*,
                                                  double*),
                                        std::vector<double> x) {
  double fx;
  f(&x, &fx);
  std::vector<double> gx(x.size());
  std::vector<double> x_prime(x.size());
  double dummy, unit = 1;

  __enzyme_autodiff(f,
                    enzyme_dup, &x, &gx,
                    enzyme_dupnoneed, &dummy, &unit);

  int i = 0;
  double eta = 1e-5 * 2; // ???

  while (true) {
    if (magnitude(gx) <= 1e-5) {
      return x;
    } else if (i == 10) {
      eta *= 2;
      i = 0;
    } else {
      for (int j = 0; j < x.size(); j++) {
        x_prime[j] = x[j] - eta * gx[j];
      }
      if (vector_dist(x, x_prime) <= 1e-5) {
        return x;
      } else {
        double fx_prime;
        f(&x_prime, &fx_prime);
        if (fx_prime < fx) {
          x = x_prime;
          fx = fx_prime;
          __enzyme_autodiff(f,
                            enzyme_dup, &x, &gx,
                            enzyme_dupnoneed, &dummy, &unit);
          i++;
        } else {
          eta /= 2;
          i = 0;
        }
      }
    }
  }
}

void accel_wrap(const std::vector<Point<double>>* charges,
                const Point<double>* px,
                double* out) {
  *out = accel(*charges, *px);
}

double naive_euler(double w) {
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

void naive_euler_wrap(const std::vector<double>* v, double* out) {
  *out = naive_euler((*v)[0]);
}

class FF : public Function<particle::Input, particle::Output> {
public:
  FF(particle::Input& input) : Function(input) {}

  void compute(particle::Output& output) {
    output = multivariate_argmin(naive_euler_wrap,
                                 std::vector<double>{_input.w0})[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"rr", function_main<FF>}
    });
}

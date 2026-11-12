#include "gradbench/evals/particle.hpp"
#include "gradbench/gd.hpp"
#include "gradbench/main.hpp"
#include "gradient.hpp"
#include <type_traits>

// Potential at (x1, x2) of the charges at (10, 10 - w) and (10, 0).
template <typename A>
A potential(A const& x1, A const& x2, A const& w) {
  return 1.0 / sqrt((x1 - 10.0) * (x1 - 10.0) +
                    (x2 - (10.0 - w)) * (x2 - (10.0 - w))) +
         1.0 / sqrt((x1 - 10.0) * (x1 - 10.0) + x2 * x2);
}

// Euler steps until the particle crosses the x axis; returns the squared x
// where it does. The force is the potential's gradient from `grad`.
template <typename T, typename Grad>
T naive_euler(T const& w, Grad const& grad) {
  auto p = [&w](auto const* x) {
    using A = std::decay_t<decltype(*x)>;
    return potential(x[0], x[1], constant<A>(w));
  };
  const double dt   = 0.1;
  T            x[2] = {0.0, 8.0};
  T            v[2] = {0.75, 0.0};
  while (true) {
    T dp[2];
    grad.gradient(p, x, dp);
    x[0] += dt * v[0];
    x[1] += dt * v[1];
    if (x[1] <= 0.0) {
      T dt_f = -x[1] / v[1];
      T x_f  = x[0] + dt_f * v[0];
      return x_f * x_f;
    }
    v[0] -= dt * dp[0];
    v[1] -= dt * dp[1];
  }
}

// Outer is the argmin's mode, Inner the Euler method's.
template <template <class, size_t> class Outer,
          template <class, size_t> class Inner>
class Particle : public Function<particle::Input, particle::Output> {
  struct Cost {
    using AD = typename Outer<double, 1>::active_type;

    Outer<double, 1> outer;
    Inner<double, 2> force;         // in objective
    Inner<AD, 2>     nested_force;  // in gradient, under outer

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      *out = naive_euler(w[0], force);
    }

    void gradient(double const* w, double* out) const {
      outer.gradient(
          [this](AD const* v) { return naive_euler(v[0], nested_force); }, w,
          out);
    }
  };

  Cost _cost;

public:
  Particle(particle::Input& input) : Function(input) {}

  void compute(particle::Output& output) {
    output = multivariate_argmin(_cost, &_input.w0)[0];
  }
};

// The potential is costly enough that one vector sweep beats two.
template <typename T, size_t N>
using ForwardVector = Forward<T, N, N>;

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"rr", function_main<Particle<Reverse, Reverse>>},
                       {"rf", function_main<Particle<Reverse, ForwardVector>>},
                       {"ff", function_main<Particle<Forward, ForwardVector>>},
                       {"fr", function_main<Particle<Forward, Reverse>>}});
}

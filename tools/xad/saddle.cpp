#include "gradbench/evals/saddle.hpp"
#include "gradbench/gd.hpp"
#include "gradbench/main.hpp"
#include "gradient.hpp"
#include <type_traits>

// saddle::objective with its arguments by reference, as copying a nested
// active is not free.
template <typename A>
A f_pq(A const& p0, A const& p1, A const& q0, A const& q1) {
  return (p0 * p0 + p1 * p1) - (q0 * q0 + q1 * q1);
}

// f(p, q) for a fixed p, as a function of q for gradient ascent; its
// gradient comes from `grad`.
template <typename T, typename Grad>
struct FixedP {
  T const*    p;
  Grad const& grad;

  size_t input_size() const { return 2; }

  void objective(T const* q, T* out) const {
    *out = f_pq(p[0], p[1], q[0], q[1]);
  }

  void gradient(T const* q, T* out) const {
    grad.gradient(
        [this](auto const* a) {
          using A = std::decay_t<decltype(*a)>;
          return f_pq(constant<A>(p[0]), constant<A>(p[1]), a[0], a[1]);
        },
        q, out);
  }
};

// max over q of f(p, q), by gradient ascent from start.
template <typename T, typename Grad>
T max_f(T const* p, double const* start, Grad const& grad) {
  T                    q0[2] = {start[0], start[1]};
  FixedP<T, Grad>      f{p, grad};
  std::vector<T> const q = multivariate_argmax(f, q0);
  return f_pq(p[0], p[1], q[0], q[1]);
}

// Outer is the argmin's mode, Inner the argmax's.
template <template <class, size_t> class Outer,
          template <class, size_t> class Inner>
class Saddle : public Function<saddle::Input, saddle::Output> {
  struct Cost {
    using AD = typename Outer<double, 2>::active_type;

    explicit Cost(double const* s) : start(s) {}

    double const*    start;
    Outer<double, 2> outer;
    Inner<double, 2> grad;         // in objective, and for the final argmax
    Inner<AD, 2>     nested_grad;  // in gradient, under outer

    size_t input_size() const { return 2; }

    void objective(double const* p, double* out) const {
      *out = max_f(p, start, grad);
    }

    void gradient(double const* p, double* out) const {
      outer.gradient(
          [this](AD const* a) { return max_f(a, start, nested_grad); }, p, out);
    }
  };

  Cost _cost;

public:
  Saddle(saddle::Input& input) : Function(input), _cost(input.start) {}

  void compute(saddle::Output& output) {
    std::vector<double> p = multivariate_argmin(_cost, _input.start);
    FixedP<double, Inner<double, 2>> f{p.data(), _cost.grad};
    std::vector<double>              q = multivariate_argmax(f, _input.start);
    output                             = {p[0], p[1], q[0], q[1]};
  }
};

// ff carries both outer directions through one inner ascent. fr cannot: the
// XAD library has no tape over a vector forward type by default.
int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"rr", function_main<Saddle<Reverse, Reverse>>},
                       {"rf", function_main<Saddle<Reverse, Forward>>},
                       {"ff", function_main<Saddle<ForwardVector, Forward>>},
                       {"fr", function_main<Saddle<Forward, Reverse>>}});
}

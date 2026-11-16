// Gradients of a scalar function of N inputs of type T, in forward or adjoint
// mode. T is double or itself an XAD active type, so the two drivers nest:
// particle and saddle use one inside the other's function.
#pragma once

#include <XAD/XAD.hpp>
#include <array>
#include <cstddef>

// An active A with value v. A's constructors take only the innermost double,
// so this also covers a value that is active itself.
template <typename A, typename T>
A constant(T const& v) {
  A a;
  xad::value(a) = v;
  return a;
}

// Forward mode, W directions per sweep: N / W sweeps, W either 1 or N.
// Vector mode (W = N) pays when the function is expensive to evaluate, but not
// for saddle's inner gradient, where f is cheap.
template <typename T, size_t N, size_t W = 1>
struct Forward {
  static_assert(W == 1 || W == N, "W is 1 or N");
  using active_type = typename xad::fwd<T, W>::active_type;

  template <typename F>
  void gradient(F const& f, T const* in, T* out) const {
    std::array<active_type, N> x;
    for (size_t i = 0; i < N; i++) {
      x[i] = constant<active_type>(in[i]);
    }
    for (size_t s = 0; s < N; s += W) {
      for (size_t i = s; i < s + W; i++) {
        direction(x[i], i - s) = 1.0;
      }
      active_type y = f(x.data());
      for (size_t i = s; i < s + W; i++) {
        out[i]                 = direction(y, i - s);
        direction(x[i], i - s) = 0.0;
      }
    }
  }

private:
  // With W = 1 the derivative is a T, not a vector of them.
  static T& direction(active_type& a, size_t d) {
    if constexpr (W == 1) {
      return xad::derivative(a);
    } else {
      return xad::derivative(a)[d];
    }
  }
};

template <typename T, size_t N>
using ForwardVector = Forward<T, N, N>;

// One adjoint sweep, on a tape kept across calls.
template <typename T, size_t N>
class Reverse {
public:
  using mode        = xad::adj<T>;
  using active_type = typename mode::active_type;

  template <typename F>
  void gradient(F const& f, T const* in, T* out) const {
    // Declared first, so the actives below are destroyed while it is active.
    Active active(_tape);
    _tape.clearAll();
    std::array<active_type, N> x;
    for (size_t i = 0; i < N; i++) {
      x[i] = constant<active_type>(in[i]);
    }
    _tape.registerInputs(x.begin(), x.end());
    _tape.newRecording();

    active_type y = f(x.data());
    _tape.registerOutput(y);
    xad::derivative(y) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < N; i++) {
      out[i] = xad::derivative(x[i]);
    }
  }

private:
  using tape_type = typename mode::tape_type;

  // Only one tape per type can be active, and the objective and gradient
  // drivers of a variant may share a tape type, so each is active only
  // during its own sweep.
  struct Active {
    tape_type& tape;
    explicit Active(tape_type& t) : tape(t) { tape.activate(); }
    ~Active() { tape.deactivate(); }
  };

  mutable tape_type _tape{false};
};

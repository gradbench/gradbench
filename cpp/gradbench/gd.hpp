// Multivariate gradient descent solver used for the particle and
// saddle evals. Since this solver is used for multiple C++-based
// tools, it has to live here.
//
// While you can use it for other evals, it may not be a particularly
// good general-purpose solver - it was designed for the needs of
// particle and eval. This also means that we should not make it more
// sophisticated, as that might undermine its use in those evals. If
// you find yourself in need of a better GD solver, then write a new
// one.

#pragma once

#include <cmath>
#include <vector>

template <typename T>
T magnitude_squared(const std::vector<T>& v) {
  T acc = 0.0;
  for (auto& x : v) {
    acc += x * x;
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
  for (size_t i = 0; i < u.size(); i++) {
    acc += (u[i] - v[i]) * (u[i] - v[i]);
  }
  return sqrt(acc);
}

/**
 * @brief Finds the multivariate argmin of a function via gradient descent.
 * @tparam FUNC A function type that has a member functions
 *  - size_t input_size()
 *  - void objective(double const * in, double* out), which stores the objective
 * function result in out
 *  - void gradient(double const * in, double* out), which stores the gradient
 * in out
 * @param T The numeric type to use (default double)
 * @param xp The starting input.
 */
template <typename FUNC, typename T = double>
std::vector<T> multivariate_argmin(FUNC const& f, T const* xp) {
  T              fx;
  std::vector<T> x(f.input_size());
  std::vector<T> gx(f.input_size());
  std::vector<T> x_prime(f.input_size());

  for (size_t j = 0; j < f.input_size(); j++) {
    x[j] = xp[j];
  }

  f.objective(xp, &fx);
  f.gradient(x.data(), gx.data());

  int    i   = 0;
  double eta = 1e-5;

  while (true) {
    if (magnitude(gx) <= 1e-5) {
      return x;
    } else if (i == 10) {
      eta *= 2;
      i = 0;
    } else {
      for (size_t j = 0; j < f.input_size(); j++) {
        x_prime[j] = x[j] - eta * gx[j];
      }
      if (vector_dist(x, x_prime) <= 1e-5) {
        return x;
      } else {
        T fx_prime;
        f.objective(x_prime.data(), &fx_prime);
        if (fx_prime < fx) {
          x  = x_prime;
          fx = fx_prime;
          f.gradient(x.data(), gx.data());
          i++;
        } else {
          eta /= 2;
          i = 0;
        }
      }
    }
  }
}

/**
 * @brief Finds the multivariate argmax of a function via gradient descent.
 * @tparam FUNC A function type that has a member functions
 *  - size_t input_size()
 *  - void objective(double const * in, double* out), which stores the objective
 * function result in out
 *  - void gradient(double const * in, double* out), which stores the gradient
 * in out
 * @tparam T The numeric type to use (default double)
 * @param xp The starting input.
 */
template <typename FUNC, typename T = double>
std::vector<T> multivariate_argmax(const FUNC& f, T const* xp) {
  struct multiplicative_inverse {
    FUNC const& _f;

    multiplicative_inverse(FUNC const& f) : _f(f) {}

    void objective(T const* x, T* out) const {
      T tmp;
      _f.objective(x, &tmp);
      *out = -tmp;
    }
    void gradient(T const* x, T* out) const {
      _f.gradient(x, out);
      for (size_t i = 0; i < _f.input_size(); i++) {
        out[i] *= -1;
      }
    }
    size_t input_size() const { return _f.input_size(); }
  };
  return multivariate_argmin(multiplicative_inverse(f), xp);
}

template <typename F>
double multivariate_max(const F& f, const double* x) {
  double              res;
  std::vector<double> xmax = multivariate_argmax(f, x);
  f.objective(xmax.data(), &res);
  return res;
}

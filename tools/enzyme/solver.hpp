#pragma once

// This is the multivariate gradient descent solver used for the
// particle and saddle evals.

#include <cmath>
#include <vector>

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

template <typename F>
std::vector<double> multivariate_argmin(const F f, const double* xp) {
  double fx;
  std::vector<double> x(f.input_size());
  std::vector<double> gx(f.input_size());
  std::vector<double> x_prime(f.input_size());

  for (int j = 0; j < f.input_size(); j++) {
    x[j] = xp[j];
  }

  f.objective(xp, &fx);
  f.gradient(x.data(), gx.data());

  int i = 0;
  double eta = 1e-5;

  while (true) {
    if (magnitude(gx) <= 1e-5) {
      return x;
    } else if (i == 10) {
      eta *= 2;
      i = 0;
    } else {
      for (int j = 0; j < f.input_size(); j++) {
        x_prime[j] = x[j] - eta * gx[j];
      }
      if (vector_dist(x, x_prime) <= 1e-5) {
        return x;
      } else {
        double fx_prime;
        f.objective(x_prime.data(), &fx_prime);
        if (fx_prime < fx) {
          x = x_prime;
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

template <typename F>
std::vector<double> multivariate_argmax(const F& f, const double* x) {
  struct C {
    const F& _f;

    C(const F& f) : _f(f) {}

    void objective(const double* x, double* out) const {
      double tmp;
      _f.objective(x, &tmp);
      *out = -tmp;
    }
    void gradient(const double* x, double* out) const {
      _f.gradient(x, out);
      for (int i = 0; i < _f.input_size(); i++) {
        out[i] *= -1;
      }
    }
    size_t input_size() const { return _f.input_size(); }
  };
  return multivariate_argmin(C(f), x);
}

template <typename F>
double multivariate_max(const F& f, const double* x) {
  double res;
  std::vector<double> xmax = multivariate_argmax(f,x);
  f.objective(xmax.data(), &res);
  return res;
}

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
std::vector<double> multivariate_argmin(const F f, std::vector<double> x) {
  double fx;
  F::objective(x.data(), &fx);
  std::vector<double> gx(x.size());
  std::vector<double> x_prime(x.size());

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
      for (int j = 0; j < x.size(); j++) {
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
std::vector<double> multivariate_argmax(std::vector<double> x) {
  struct C {
    double cost(const std::vector<double>& x) {
      return -F::cost(x);
    }
    std::vector<double> gradient(const std::vector<double>& x) {
      std::vector<double> r = F::gradient(x);
      for (auto &x : r) {
        x = -x;
      }
      return r;
    }
  };
  return multivariate_argmin<C>(C(), x);
}

template <typename F>
std::vector<double> multivariate_max(const std::vector<double>& x) {
  return F::cost(multivariate_argmax<F>(x));
}

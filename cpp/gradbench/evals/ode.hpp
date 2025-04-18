// Based on
//
// https://github.com/bradbell/cmpad/blob/e375e0606f9b6f6769ea4ce0a57a00463a090539/cpp/include/cmpad/algo/runge_kutta.hpp
//
// originally by Bradley M. Bell <bradbell@seanet.com>, and used here
// under the terms of the EPL-2.0 or GPL-2.0-or-later.
//
// The implementation in cmpad is factored into a generic Runge-Kutta
// module and an instantiation for the specific function under
// consideration. In this implementation, this is all inlined for
// implementation simplicity. This is generally written in a more
// C-like way for the benefit of tools that are not so good at C++.

#include <cassert>
#include <vector>

#include "gradbench/main.hpp"
#include "json.hpp"

namespace ode {

struct Input {
  std::vector<double> x;
  size_t              s;
};

typedef std::vector<double> PrimalOutput;

typedef std::vector<double> GradientOutput;

template <typename T>
void ode_fun(size_t n, const T* __restrict__ x, const T* __restrict__ y,
             T* __restrict__ z) {
  z[0] = x[0];
#pragma omp parallel for
  for (size_t i = 1; i < n; i++) {
    z[i] = x[i] * y[i - 1];
  }
}

template <typename T>
void primal(size_t n, const T* __restrict__ xi, size_t s, T* __restrict__ yf) {
  T              tf = T(2);
  T              h  = tf / T(s);
  std::vector<T> k1(n), k2(n), k3(n), k4(n), y_tmp(n);
  std::fill(yf, yf + n, T(0));

  for (size_t i_step = 0; i_step < s; i_step++) {
    ode_fun(n, xi, yf, k1.data());

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      y_tmp[i] = yf[i] + h * k1[i] / T(2);
    }
    ode_fun(n, xi, y_tmp.data(), k2.data());

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      y_tmp[i] = yf[i] + h * k2[i] / T(2);
    }
    ode_fun(n, xi, y_tmp.data(), k3.data());

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      y_tmp[i] = yf[i] + h * k3[i];
    }
    ode_fun(n, xi, y_tmp.data(), k4.data());

    for (size_t i = 0; i < n; i++) {
      yf[i] += h * (k1[i] + T(2) * k2[i] + T(2) * k3[i] + k4[i]) / T(6);
    }
  }
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.x = j["x"].get<std::vector<double>>();
  p.s = j["s"].get<size_t>();
}

class Primal : public Function<Input, PrimalOutput> {
public:
  Primal(Input& input) : Function(input) {}

  void compute(PrimalOutput& output) {
    output.resize(_input.x.size());
    size_t n = _input.x.size();
    primal<double>(n, _input.x.data(), _input.s, output.data());
  }
};

}  // namespace ode

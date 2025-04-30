#include <vector>

#include "gradbench/main.hpp"
#include "gradbench/multithread.hpp"
#include "json.hpp"

namespace llsq {
struct Input {
  std::vector<double> x;
  size_t              n;  // Size of 't' and 's'.
};

typedef double PrimalOutput;

typedef std::vector<double> GradientOutput;

double t(double i, double n) { return -1 + i * 2 / (n - 1); }

template <typename T>
T s(T ti) {
  return (T(0) < ti) - (ti < T(0));
}

template <typename T>
void primal(size_t n, size_t m, const T* __restrict__ x, T* __restrict__ out) {
  T sum = T(0);
  for (size_t i = 0; i < n; i++) {
    T ti        = t(i, n);
    T inner_sum = s(ti);
    T mul       = 1;
    for (size_t j = 0; j < m; j++) {
      inner_sum -= x[j] * mul;
      mul *= ti;
    }
    sum += inner_sum * inner_sum;
  }
  *out = sum / T(2);
}

// Multithreaded version.
template <>
void primal(size_t n, size_t m, const double* __restrict__ x,
            double* __restrict__ out) {
  std::vector<double> sums(num_threads());
#pragma omp parallel
  {
    double sum = 0;
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      double ti        = t(i, n);
      double inner_sum = s(ti);
      double mul       = 1;
      for (size_t j = 0; j < m; j++) {
        inner_sum -= x[j] * mul;
        mul *= ti;
      }
      sum += inner_sum * inner_sum;
    }
    sums[thread_num()] = sum;
  }
  double sum = 0;
  for (auto x : sums) {
    sum += x;
  }
  *out = sum / double(2);
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.x = j["x"].get<std::vector<double>>();
  p.n = j["n"].get<size_t>();
}

class Primal : public Function<Input, PrimalOutput> {
public:
  Primal(Input& input) : Function(input) {}

  void compute(PrimalOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    primal<double>(n, m, _input.x.data(), &output);
  }
};

}  // namespace llsq

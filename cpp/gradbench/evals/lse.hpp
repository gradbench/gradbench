#include <cmath>
#include <vector>

#include "gradbench/main.hpp"
#include "gradbench/multithread.hpp"
#include "json.hpp"

namespace lse {

struct Input {
  std::vector<double> x;
};

typedef double              PrimalOutput;
typedef std::vector<double> GradientOutput;

template <typename T>
void primal(size_t n, const T* __restrict__ x, T* __restrict__ out) {
  T A = x[0];
  for (size_t i = 1; i < n; i++) {
    A = std::max(A, x[i]);
  }

  T s = 0;
  for (size_t i = 0; i < n; i++) {
    s += exp(x[i] - A);
  }

  *out = log(s) + A;
}

template <>
void primal(size_t n, const double* __restrict__ x, double* __restrict__ out) {
  std::vector<double> As(num_threads());
#pragma omp parallel
  {
    double priv_A = x[0];
#pragma omp for
    for (size_t i = 1; i < n; i++) {
      priv_A = std::max(priv_A, x[i]);
    }
    As[thread_num()] = priv_A;
  }
  double A = 0;
  for (int i = 0; i < num_threads(); i++) {
    A = std::max(A, As[i]);
  }

  std::vector<double> Ss(num_threads());
#pragma omp parallel
  {
    double priv_s = 0;
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      priv_s += exp(x[i] - A);
    }
    Ss[thread_num()] = priv_s;
  }
  double s = 0;
  for (int i = 0; i < num_threads(); i++) {
    s += Ss[i];
  }

  *out = log(s) + A;
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.x = j["x"].get<std::vector<double>>();
}

class Primal : public Function<Input, PrimalOutput> {
public:
  Primal(Input& input) : Function(input) {}

  void compute(PrimalOutput& output) {
    size_t n = _input.x.size();
    primal<double>(n, _input.x.data(), &output);
  }
};

}  // namespace lse

#include <cmath>
#include <vector>

#include "gradbench/main.hpp"
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

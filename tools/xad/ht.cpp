#include "gradbench/evals/ht.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <algorithm>
#include <vector>

// Dense Jacobian by vector forward mode: each pass propagates N tangent
// directions (Jacobian columns) at once.
class Jacobian : public Function<ht::Input, ht::JacOutput> {
  static constexpr int N = 4;
  using AD               = xad::fwd<double, N>::active_type;

  bool _complicated;

public:
  Jacobian(ht::Input& input)
      : Function(input), _complicated(input.us.size() != 0) {}

  void compute(ht::JacOutput& output) {
    const int nrows       = 3 * _input.data.correspondences.size();
    const int off         = _complicated ? 2 : 0;
    const int ncols       = off + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = nrows;
    output.jacobian.resize(nrows * ncols);

    std::vector<AD> theta(_input.theta.begin(), _input.theta.end());
    std::vector<AD> us(_input.us.begin(), _input.us.end());
    std::vector<AD> err(nrows);

    // Seed (or, with v = 0, clear) column c in direction d. In the
    // complicated variant the us part is compressed into two columns:
    // column 0 seeds the odd-index us, column 1 the even-index us.
    auto seed = [&](int c, int d, double v) {
      if (c < off) {
        for (size_t j = 0; j < us.size(); j++) {
          xad::derivative(us[j])[d] = v * ((1 + c + j) % 2);
        }
      } else {
        xad::derivative(theta[c - off])[d] = v;
      }
    };

    for (int c0 = 0; c0 < ncols; c0 += N) {
      const int dirs = std::min(N, ncols - c0);
      for (int d = 0; d < dirs; d++) {
        seed(c0 + d, d, 1.0);
      }

      if (_complicated) {
        ht::objective(theta.data(), us.data(), &_input.data, err.data());
      } else {
        ht::objective(theta.data(), &_input.data, err.data());
      }

      for (int d = 0; d < dirs; d++) {
        for (int j = 0; j < nrows; j++) {
          output.jacobian[(c0 + d) * nrows + j] = xad::derivative(err[j])[d];
        }
        seed(c0 + d, d, 0.0);
      }
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}

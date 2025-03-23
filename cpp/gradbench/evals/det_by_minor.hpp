// Based on
//
// https://github.com/bradbell/cmpad/blob/e375e0606f9b6f6769ea4ce0a57a00463a090539/cpp/include/cmpad/algo/det_of_minor.hpp
//
// originally by Bradley M. Bell <bradbell@seanet.com>, and used here
// under the terms of the EPL-2.0 or GPL-2.0-or-later.

#include <vector>
#include <cassert>
#include "json.hpp"

namespace det_by_minor {

struct Input {
  std::vector<double> A;
  size_t ell;
};

typedef double PrimalOutput;

typedef std::vector<double> GradientOutput;

// r[i] is the row following row 'i', and similarly 'c[i]' for
// columns. This is used to (relatively) efficiently remove rows and
// columns, without actually moving the data.
template <typename T>
T det_of_minor(const T* __restrict__ A,
               size_t n,
               size_t m,
               std::vector<size_t> &r,
               std::vector<size_t> &c) {
  size_t R0 = r[n];
  assert(R0 < n);
  size_t Cj = c[n];
  assert(Cj < n);

  if (m == 1) {
    return A[R0 * n + Cj];
  }

  // Determinant of the minor M.
  T detM(0);
  // Sign of factor for next sub-minor.
  int sign = 1;
  // Remove row with index 0 in M from all the sub-minors of M
  r[n] = r[R0];
  // C(j-1): initial index in c for previous column of the minor M
  size_t Cj1 = n;

  // for each column of M
  for (size_t j = 0; j < m; j++) {
    // M[0,j] = A[ R0, Cj ]
    // element with index (0, j) in the minor M
    T M0j = A[R0 * n + Cj];

    // remove column with index j in M to form next sub-minor S of M
    c[Cj1] = c[Cj];

    // detS: compute determinant of S, the sub-minor of M with row
    // R(0) and column C(j) removed.
    T detS = det_of_minor(A, n, m - 1, r, c);

    // restore column with index j in represenation of M as a minor of
    // A
    c[Cj1] = Cj;

    // detM: include this sub-minor term in the summation
    if (sign > 0) {
      detM = detM + M0j * detS;
    } else {
      detM = detM - M0j * detS;
    }

    // advance to next column of M
    Cj1  = Cj;
    Cj   = c[Cj];
    sign = -sign;
  }

  // restore row zero to the minor representation for M
  r[n] = R0;

  return detM;
}

template<typename T>
void primal(size_t ell,
            const T* __restrict__ A,
            T* __restrict__ out) {
  std::vector<size_t> r(ell + 1);
  std::vector<size_t> c(ell + 1);
  for(size_t i = 0; i < ell; i++) {
    r[i] = i+1;
    c[i] = i+1;
  }
  *out = det_of_minor(A, ell, ell, r, c);
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.A = j["A"].get<std::vector<double>>();
  p.ell = j["ell"].get<size_t>();
}

class Primal : public Function<Input, PrimalOutput> {
public:
  Primal(Input& input) : Function(input) {}

  void compute(PrimalOutput& output) {
    primal<double>(_input.ell,
                   _input.A.data(),
                   &output);
  }
};

}

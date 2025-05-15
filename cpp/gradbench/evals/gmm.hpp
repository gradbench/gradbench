// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Derived from
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/gmm.h
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/GMMData.h

#pragma once

#include <vector>

#include "adbench/shared/defs.h"
#include "adbench/shared/matrix.h"
#include "gradbench/main.hpp"

namespace gmm {
//// Declarations

typedef ::Wishart Wishart;

template <typename T>
void objective(int d, int k, int n, const T* const alpha, const T* const mu,
               const T* const q, const T* const l, const double* const x,
               Wishart wishart, T* err);

template <typename T>
T logsumexp(int n, const T* const x);

// p: dim
// k: number of components
// wishart parameters
// sum_qs: k sums of log diags of Qs
// Qdiags: d*k
// icf: (p*(p+1)/2)*k inverse covariance factors
template <typename T>
T log_wishart_prior(int p, int k, Wishart wishart, const T* const sum_qs,
                    const T* const Qdiags, const T* const icf);

template <typename T>
void Qtimesx(int d, const T* const Qdiag,
             const T* const ltri,  // strictly lower triangular part
             const T* const x, T* out);

//// Definitions

template <typename T>
T logsumexp(int n, const T* const x) {
  T mx   = arr_max(n, x);
  T semx = 0.;
  for (int i = 0; i < n; i++) {
    semx = semx + exp(x[i] - mx);
  }
  return log(semx) + mx;
}

template <typename T>
T log_wishart_prior(int d, int k, const Wishart wishart, const T* const sum_qs,
                    const T* const Qdiags, const T* const l) {
  int       n    = d + wishart.m + 1;
  const int l_sz = d * (d - 1) / 2;

  double C = n * d * (log(wishart.gamma) - 0.5 * log(2)) -
             log_gamma_distrib(0.5 * n, d);

  T out = 0;
  for (int ik = 0; ik < k; ik++) {
    T frobenius = sqnorm(d, &Qdiags[ik * d]) + sqnorm(l_sz, &l[ik * l_sz]);
    out += 0.5 * wishart.gamma * wishart.gamma * frobenius -
           wishart.m * sum_qs[ik];
  }

  return -out + k * C;
}

template <typename T>
void Qtimesx(int d, const T* const Qdiag,
             const T* const ltri,  // strictly lower triangular part
             const T* const x, T* out) {
  for (int id = 0; id < d; id++) {
    out[id] = Qdiag[id] * x[id];
  }

  int Lparamsidx = 0;
  for (int i = 0; i < d; i++) {
    for (int j = i + 1; j < d; j++) {
      out[j] = out[j] + ltri[Lparamsidx] * x[i];
      Lparamsidx++;
    }
  }
}

template <typename T>
void objective(int d, int k, int n, const T* __restrict__ const alpha,
               const T* __restrict__ const mu, const T* __restrict__ const q,
               const T* __restrict__ const l,
               const double* __restrict__ const x, Wishart wishart,
               T* __restrict__ err) {
  const double CONSTANT = -n * d * 0.5 * log(2 * M_PI);
  const int    l_sz     = d * (d - 1) / 2;

  std::vector<T> Qdiags(d * k);
  std::vector<T> sum_qs(k);

  for (int i = 0; i < d * k; i++) {
    Qdiags[i] = exp(q[i]);
  }
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      sum_qs[i] += q[i * d + j];
    }
  }

  std::vector<T> xcentered(d);
  std::vector<T> Qxcentered(d);
  std::vector<T> main_term(k);
  T              slse = 0.;

#ifdef USE_OPENMP
#pragma omp parallel for reduction(+ : slse)                                   \
    firstprivate(xcentered, Qxcentered, main_term)
#endif
  for (int ix = 0; ix < n; ix++) {
    for (int ik = 0; ik < k; ik++) {
      subtract(d, &x[ix * d], &mu[ik * d], &xcentered[0]);
      Qtimesx(d, &Qdiags[ik * d], &l[ik * l_sz], &xcentered[0], &Qxcentered[0]);

      main_term[ik] = alpha[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
    }
    T lsum = logsumexp(k, &main_term[0]);
    slse += lsum;
  }

  T lse_alpha = logsumexp(k, alpha);

  *err = CONSTANT + slse - n * lse_alpha;

  T ws = log_wishart_prior(d, k, wishart, &sum_qs[0], &Qdiags[0], l);

  *err = *err + ws;
}

//// Interface

struct Input {
  int                 d, k, n;
  std::vector<double> alpha, mu, q, l, x;
  Wishart             wishart;
};

typedef double ObjOutput;

struct JacOutput {
  int                 d, k, n;
  std::vector<double> alpha;
  std::vector<double> mu, q, l;
};

using json = nlohmann::json;

static void from_json(const json& j, Input& p) {
  p.d      = j["d"].get<int>();
  p.k      = j["k"].get<int>();
  p.n      = j["n"].get<int>();
  p.alpha = j["alpha"].get<std::vector<double>>();

  auto mu = j["mu"].get<std::vector<std::vector<double>>>();
  auto q  = j["q"].get<std::vector<std::vector<double>>>();
  auto l  = j["l"].get<std::vector<std::vector<double>>>();
  auto x  = j["x"].get<std::vector<std::vector<double>>>();
  for (int i = 0; i < p.k; i++) {
    p.mu.insert(p.mu.end(), mu[i].begin(), mu[i].end());
    p.q.insert(p.q.end(), q[i].begin(), q[i].end());
    p.l.insert(p.l.end(), l[i].begin(), l[i].end());
  }
  for (int i = 0; i < p.n; i++) {
    p.x.insert(p.x.end(), x[i].begin(), x[i].end());
  }

  p.wishart.gamma = j["gamma"].get<double>();
  p.wishart.m     = j["m"].get<int>();
}

static void to_json(nlohmann::json& j, const JacOutput& p) {
  const int                        l_sz = p.d * (p.d - 1) / 2;
  std::vector<std::vector<double>> mu(p.k), q(p.k), l(p.k);

  for (int i = 0; i < p.k; i++) {
    mu[i].insert(mu[i].end(), p.mu.begin() + i * p.d,
                 p.mu.begin() + (i + 1) * p.d);
    q[i].insert(q[i].end(), p.q.begin() + i * p.d, p.q.begin() + (i + 1) * p.d);
    l[i].insert(l[i].end(), p.l.begin() + i * l_sz,
                p.l.begin() + (i + 1) * l_sz);
  }

  j = {{"mu", mu}, {"q", q}, {"l", l}, {"alpha", p.alpha}};
}

class Objective : public Function<Input, ObjOutput> {
public:
  Objective(Input& input) : Function(input) {}

  void compute(ObjOutput& output) {
    objective(_input.d, _input.k, _input.n, _input.alpha.data(),
              _input.mu.data(), _input.q.data(), _input.l.data(),
              _input.x.data(), _input.wishart, &output);
  }
};

}  // namespace gmm

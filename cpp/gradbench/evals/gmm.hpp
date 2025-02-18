// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Derived from
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/gmm.h
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/GMMData.h

#pragma once

#include <vector>
#include "adbench/shared/matrix.h"
#include "adbench/shared/defs.h"

namespace gmm {
//// Declarations

typedef ::Wishart Wishart;

// d: dim
// k: number of gaussians
// n: number of points
// alphas: k logs of mixture weights (unnormalized), so
//          weights = exp(log_alphas) / sum(exp(log_alphas))
// means: d*k component means
// icf: (d*(d+1)/2)*k inverse covariance factors 
//                  every icf entry stores firstly log of diagonal and then 
//          columnwise other entris
//          To generate icf in MATLAB given covariance C :
//              L = inv(chol(C, 'lower'));
//              inv_cov_factor = [log(diag(L)); L(au_tril_indices(d, -1))]
// wishart: wishart distribution parameters
// x: d*n points
// err: 1 output
template<typename T>
void objective(int d, int k, int n, const T* const alphas, const T* const means,
               const T* const icf, const double* const x, Wishart wishart, T* err);

template<typename T>
T logsumexp(int n, const T* const x);

// p: dim
// k: number of components
// wishart parameters
// sum_qs: k sums of log diags of Qs
// Qdiags: d*k
// icf: (p*(p+1)/2)*k inverse covariance factors
template<typename T>
T log_wishart_prior(int p, int k,
                    Wishart wishart,
                    const T* const sum_qs,
                    const T* const Qdiags,
                    const T* const icf);

template<typename T>
void preprocess_qs(int d, int k,
                   const T* const icf,
                   T* sum_qs,
                   T* Qdiags);

template<typename T>
void Qtimesx(int d,
             const T* const Qdiag,
             const T* const ltri, // strictly lower triangular part
             const T* const x,
             T* out);

//// Definitions

template<typename T>
T logsumexp(int n, const T* const x)
{
  T mx = arr_max(n, x);
  T semx = 0.;
  for (int i = 0; i < n; i++)
    {
      semx = semx + exp(x[i] - mx);
    }
  return log(semx) + mx;
}

template<typename T>
T log_wishart_prior(int p, int k,
                    Wishart wishart,
                    const T* const sum_qs,
                    const T* const Qdiags,
                    const T* const icf)
{
  int n = p + wishart.m + 1;
  int icf_sz = p * (p + 1) / 2;

  double C = n * p * (log(wishart.gamma) - 0.5 * log(2)) - log_gamma_distrib(0.5 * n, p);

  T out = 0;
  for (int ik = 0; ik < k; ik++)
    {
      T frobenius = sqnorm(p, &Qdiags[ik * p]) + sqnorm(icf_sz - p, &icf[ik * icf_sz + p]);
      out = out + 0.5 * wishart.gamma * wishart.gamma * (frobenius)
        -wishart.m * sum_qs[ik];
    }

  return out - k * C;
}

template<typename T>
void preprocess_qs(int d, int k,
                   const T* const icf,
                   T* sum_qs,
                   T* Qdiags)
{
  int icf_sz = d * (d + 1) / 2;
  for (int ik = 0; ik < k; ik++)
    {
      sum_qs[ik] = 0.;
      for (int id = 0; id < d; id++)
        {
          T q = icf[ik * icf_sz + id];
          sum_qs[ik] = sum_qs[ik] + q;
          Qdiags[ik * d + id] = exp(q);
        }
    }
}

template<typename T>
void Qtimesx(int d,
             const T* const Qdiag,
             const T* const ltri, // strictly lower triangular part
             const T* const x,
             T* out) {
  for (int id = 0; id < d; id++)
    out[id] = Qdiag[id] * x[id];

  int Lparamsidx = 0;
  for (int i = 0; i < d; i++)
    {
      for (int j = i + 1; j < d; j++)
        {
          out[j] = out[j] + ltri[Lparamsidx] * x[i];
          Lparamsidx++;
        }
    }
}

template<typename T>
void objective(int d, int k, int n,
               const T* __restrict__ const alphas,
               const T* __restrict__ const means,
               const T* __restrict__ const icf,
               const double* __restrict__ const x,
               Wishart wishart,
               T* __restrict__ err) {
  const double CONSTANT = -n * d * 0.5 * log(2 * M_PI);
  int icf_sz = d * (d + 1) / 2;

  std::vector<T> Qdiags(d * k);
  std::vector<T> sum_qs(k);
  std::vector<T> xcentered(d);
  std::vector<T> Qxcentered(d);
  std::vector<T> main_term(k);

  preprocess_qs(d, k, icf, &sum_qs[0], &Qdiags[0]);

  T slse = 0.;
  for (int ix = 0; ix < n; ix++)
    {
      for (int ik = 0; ik < k; ik++)
        {
          subtract(d, &x[ix * d], &means[ik * d], &xcentered[0]);
          Qtimesx(d, &Qdiags[ik * d], &icf[ik * icf_sz + d], &xcentered[0], &Qxcentered[0]);

          main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
        }
      slse = slse + logsumexp(k, &main_term[0]);
    }

  T lse_alphas = logsumexp(k, alphas);

  *err = CONSTANT + slse - n * lse_alphas;

  *err = *err + log_wishart_prior(d, k, wishart, &sum_qs[0], &Qdiags[0], icf);
}

//// Interface

struct Input {
  int d, k, n;
  std::vector<double> alphas, means, icf, x;
  Wishart wishart;
};

typedef double ObjOutput;

typedef std::vector<double> JacOutput;

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.d = j["d"].get<int>();
  p.k = j["k"].get<int>();
  p.n = j["n"].get<int>();
  p.alphas = j["alpha"].get<std::vector<double>>();

  auto means = j["means"].get<std::vector<std::vector<double>>>();
  auto icf = j["icf"].get<std::vector<std::vector<double>>>();
  auto x = j["x"].get<std::vector<std::vector<double>>>();
  for (int i = 0; i < p.k; i++) {
    p.means.insert(p.means.end(), means[i].begin(), means[i].end());
    p.icf.insert(p.icf.end(), icf[i].begin(), icf[i].end());
  }
  for (int i = 0; i < p.n; i++) {
    p.x.insert(p.x.end(), x[i].begin(), x[i].end());
  }

  p.wishart.gamma = j["gamma"].get<double>();
  p.wishart.m = j["m"].get<int>();
}

class Objective : public Function<Input, ObjOutput> {
public:
  Objective(Input& input) : Function(input) {}

  void compute(ObjOutput& output) {
    objective(_input.d, _input.k, _input.n,
              _input.alphas.data(), _input.means.data(),
              _input.icf.data(), _input.x.data(), _input.wishart,
              &output);
  }
};

}

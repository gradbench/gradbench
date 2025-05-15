// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp

#include <algorithm>

#include "gradbench/evals/gmm.hpp"
#include "gradbench/main.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  std::vector<double> _J;

public:
  Jacobian(gmm::Input& input) : Function(input) {
    int d = _input.d, n = _input.n, k = _input.k;
    int Jcols  = (k * (d + 1) * (d + 2)) / 2;

    // Construct tape.
    adouble *aalpha, *amu, *aq, *al, aerr;

    _J.resize(Jcols);

    trace_on(tapeTag);

    aalpha = new adouble[k];
    for (int i = 0; i < k; i++) {
      aalpha[i] <<= _input.alpha[i];
    }
    amu = new adouble[k * d];
    for (int i = 0; i < d * k; i++) {
      amu[i] <<= _input.mu[i];
    }
    aq = new adouble[k * d];
    for (int i = 0; i < k*d; i++) {
      aq[i] <<= _input.q[i];
    }
    al = new adouble[k * (d * (d-1)/2)];
    for (int i = 0; i < k * (d * (d-1)/2); i++) {
      al[i] <<= _input.l[i];
    }

    gmm::objective(d, k, n, aalpha, amu, aq, al, _input.x.data(),
                   _input.wishart, &aerr);

    double err;
    aerr >>= err;

    trace_off();

    delete[] aalpha;
    delete[] amu;
    delete[] aq;
    delete[] al;
  }

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;
    size_t Jcols = _J.size();

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    double* in = new double[Jcols];
    int off;

    off = 0;
    memcpy(in + off, _input.alpha.data(), _input.alpha.size() * sizeof(double));
    off += _input.alpha.size();
    memcpy(in + off, _input.mu.data(), _input.mu.size() * sizeof(double));
    off += _input.mu.size();
    memcpy(in + off, _input.q.data(), _input.q.size() * sizeof(double));
    off += _input.q.size();
    memcpy(in + off, _input.l.data(), _input.l.size() * sizeof(double));

    gradient(tapeTag, Jcols, in, _J.data());

    delete[] in;

    off = 0;
    std::copy(_J.begin() + off, _J.begin() + off + _input.alpha.size(),
              output.alpha.begin());
    off += _input.alpha.size();
    std::copy(_J.begin() + off, _J.begin() + off + _input.mu.size(),
              output.mu.begin());
    off += _input.mu.size();
    std::copy(_J.begin() + off, _J.begin() + off + _input.q.size(),
              output.q.begin());
    off += _input.q.size();
    std::copy(_J.begin() + off, _J.begin() + off + _input.l.size(),
              output.l.begin());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}

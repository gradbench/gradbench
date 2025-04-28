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
public:
  Jacobian(gmm::Input& input) : Function(input) {
    int d = _input.d, n = _input.n, k = _input.k;

    // Construct tape.
    int      icf_sz = d * (d + 1) / 2;
    adouble *aalphas, *ameans, *aicf, aerr;

    trace_on(tapeTag);

    aalphas = new adouble[k];
    for (int i = 0; i < k; i++) {
      aalphas[i] <<= _input.alphas[i];
    }
    ameans = new adouble[d * k];
    for (int i = 0; i < d * k; i++) {
      ameans[i] <<= _input.means[i];
    }
    aicf = new adouble[icf_sz * k];
    for (int i = 0; i < icf_sz * k; i++) {
      aicf[i] <<= _input.icf[i];
    }

    gmm::objective(d, k, n, aalphas, ameans, aicf, _input.x.data(),
                   _input.wishart, &aerr);

    double err;
    aerr >>= err;

    trace_off();

    delete[] aalphas;
    delete[] ameans;
    delete[] aicf;
  }

  void compute(gmm::JacOutput& output) {
    int d = _input.d, k = _input.k;
    int icf_sz = d * (d + 1) / 2;
    int Jcols  = (k * (d + 1) * (d + 2)) / 2;
    output.resize(Jcols);

    double* in = new double[Jcols];
    memcpy(in, _input.alphas.data(), k * sizeof(double));
    int off = k;
    memcpy(in + off, _input.means.data(), d * k * sizeof(double));
    off += d * k;
    memcpy(in + off, _input.icf.data(), icf_sz * k * sizeof(double));
    gradient(tapeTag, Jcols, in, output.data());
    delete[] in;
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}

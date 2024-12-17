// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp

#include "AdolCGMM.h"
#include "adbench/shared/gmm.h"

#include <string.h>
#include <iostream>

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

void AdolCGMM::prepare(GMMInput&& input) {
  int d = input.d, n = input.n, k = input.k;
  _input = input;
  int Jcols = (k * (d + 1) * (d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };

  // Construct tape.
  int icf_sz = d*(d + 1) / 2;
  adouble *aalphas, *ameans, *aicf, aerr;

  trace_on(tapeTag);

  aalphas = new adouble[k];
  for (int i = 0; i < k; i++) {
    aalphas[i] <<= input.alphas[i];
  }
  ameans = new adouble[d*k];
  for (int i = 0; i < d*k; i++) {
    ameans[i] <<= input.means[i];
  }
  aicf = new adouble[icf_sz*k];
  for (int i = 0; i < icf_sz*k; i++) {
    aicf[i] <<= input.icf[i];
  }

  gmm_objective(d, k, n, aalphas, ameans,
		aicf, input.x.data(), input.wishart, &aerr);

  aerr >>= _output.objective;

  trace_off();

  delete[] aalphas;
  delete[] ameans;
  delete[] aicf;

}

GMMOutput AdolCGMM::output() {
  return _output;
}

void AdolCGMM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                  _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
  }
}

void AdolCGMM::calculate_jacobian(int times) {
  int d = _input.d, k = _input.k;
  int icf_sz = d*(d + 1) / 2;
  int Jcols = (k * (d + 1) * (d + 2)) / 2;

  for (int i = 0; i < times; ++i) {
    double *in = new double[Jcols];
    memcpy(in, _input.alphas.data(), k * sizeof(double));
    int off = k;
    memcpy(in + off, _input.means.data(), d*k * sizeof(double));
    off += d * k;
    memcpy(in + off, _input.icf.data(), icf_sz*k * sizeof(double));
    gradient(tapeTag, Jcols, in, _output.gradient.data());
    delete[] in;
  }
}

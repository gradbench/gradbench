// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp

#include "AdolCGMM.h"
#include "adbench/shared/gmm.h"

#include <string.h>

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

AdolCGMM::AdolCGMM(GMMInput& input) : ITest(input) {
  int d = _input.d, k = _input.k;
  int Jcols = (k * (d + 1) * (d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };
}

void AdolCGMM::prepare_jacobian() {
  int d = _input.d, n = _input.n, k = _input.k;

  // Construct tape.
  int icf_sz = d*(d + 1) / 2;
  adouble *aalphas, *ameans, *aicf, aerr;

  trace_on(tapeTag);

  aalphas = new adouble[k];
  for (int i = 0; i < k; i++) {
    aalphas[i] <<= _input.alphas[i];
  }
  ameans = new adouble[d*k];
  for (int i = 0; i < d*k; i++) {
    ameans[i] <<= _input.means[i];
  }
  aicf = new adouble[icf_sz*k];
  for (int i = 0; i < icf_sz*k; i++) {
    aicf[i] <<= _input.icf[i];
  }

  gmm_objective(d, k, n, aalphas, ameans,
		aicf, _input.x.data(), _input.wishart, &aerr);

  aerr >>= _output.objective;

  trace_off();

  delete[] aalphas;
  delete[] ameans;
  delete[] aicf;
}

void AdolCGMM::calculate_objective() {
  gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
}

void AdolCGMM::calculate_jacobian() {
  int d = _input.d, k = _input.k;
  int icf_sz = d*(d + 1) / 2;
  int Jcols = (k * (d + 1) * (d + 2)) / 2;

  double *in = new double[Jcols];
  memcpy(in, _input.alphas.data(), k * sizeof(double));
  int off = k;
  memcpy(in + off, _input.means.data(), d*k * sizeof(double));
  off += d * k;
  memcpy(in + off, _input.icf.data(), icf_sz*k * sizeof(double));
  gradient(tapeTag, Jcols, in, _output.gradient.data());
  delete[] in;
}

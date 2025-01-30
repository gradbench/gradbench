// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeBA.cpp

#include "TapenadeBA.h"

TapenadeBA::TapenadeBA(BAInput& input) : ITest(input) {
  _input = input;
  _output = {
    std::vector<double>(2 * _input.p),
    std::vector<double>(_input.p),
    BASparseMat(_input.n, _input.m, _input.p)
  };

  reproj_err_d = std::vector<double>(2 * (BA_NCAMPARAMS + 3 + 1));
  reproj_err_d_row = std::vector<double>(BA_NCAMPARAMS + 3 + 1);
}

void TapenadeBA::calculate_objective() {
  ba_objective(_input.n,
               _input.m,
               _input.p,
               _input.cams.data(),
               _input.X.data(),
               _input.w.data(),
               _input.obs.data(),
               _input.feats.data(),
               _output.reproj_err.data(),
               _output.w_err.data()
               );
}

void TapenadeBA::calculate_jacobian() {
  _output.J.clear();
  calculate_reproj_error_jacobian_part();
  calculate_weight_error_jacobian_part();
}



void TapenadeBA::calculate_reproj_error_jacobian_part()
{
    double errb[2];     // stores dY
                        // (i-th element equals to 1.0 for calculating i-th jacobian row)

    double err[2];      // stores fictive _output
                        // (Tapenade doesn't calculate an original function in reverse mode)

    double* cam_gradient_part = reproj_err_d_row.data();
    double* x_gradient_part = reproj_err_d_row.data() + BA_NCAMPARAMS;
    double* weight_gradient_part = reproj_err_d_row.data() + BA_NCAMPARAMS + 3;

    for (int i = 0; i < _input.p; i++)
    {
        int camIdx = _input.obs[2 * i + 0];
        int ptIdx = _input.obs[2 * i + 1];

        // calculate first row
        errb[0] = 1.0;
        errb[1] = 0.0;
        compute_reproj_error_b(
            &_input.cams[camIdx * BA_NCAMPARAMS],
            cam_gradient_part,
            &_input.X[ptIdx * 3],
            x_gradient_part,
            &_input.w[i],
            weight_gradient_part,
            &_input.feats[i * 2],
            err,
            errb
        );

        // fill first row elements
        for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++)
        {
            reproj_err_d[2 * j] = reproj_err_d_row[j];
        }

        // calculate second row
        errb[0] = 0.0;
        errb[1] = 1.0;
        compute_reproj_error_b(
            &_input.cams[camIdx * BA_NCAMPARAMS],
            cam_gradient_part,
            &_input.X[ptIdx * 3],
            x_gradient_part,
            &_input.w[i],
            weight_gradient_part,
            &_input.feats[i * 2],
            err,
            errb
        );

        // fill second row elements
        for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++)
        {
            reproj_err_d[2 * j + 1] = reproj_err_d_row[j];
        }

        _output.J.insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
    }
}



void TapenadeBA::calculate_weight_error_jacobian_part()
{
    for (int j = 0; j < _input.p; j++)
    {
        double wb;              // stores calculated derivative

        double err = 0.0;       // stores fictive _output
                                // (Tapenade doesn't calculate an original function in reverse mode)

        double errb = 1.0;      // stores dY
                                // (equals to 1.0 for derivative calculation)

        compute_zach_weight_error_b(&_input.w[j], &wb, &err, &errb);
        _output.J.insert_w_err_block(j, wb);
    }
}

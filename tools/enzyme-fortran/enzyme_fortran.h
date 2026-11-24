// Gradients computed by Enzyme over the shared Fortran primals
// (cpp/gradbench/evals/fortran), see grad_*.f90. All arguments are by
// reference, as in Fortran; output adjoints must be zeroed by the caller.
#pragma once

extern "C" {
void lse_enzyme_gradient(const int* n, const double* x, double* xb);
void llsq_enzyme_gradient(const int* n, const int* m, const double* x,
                          double* xb);
void ode_enzyme_gradient(const int* n, const double* xi, double* xib,
                         const int* s);
void det_enzyme_gradient(const int* ell, const double* A, double* Ab);
}

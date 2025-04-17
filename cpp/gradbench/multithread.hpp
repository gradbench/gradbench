#pragma once

// Tiny wrapper library to make it slightly more convenient to write
// code that makes conditional use of OpenMP.

// Corresponds to omp_get_max_threads().
//
// Returns 1 when OpenMP is not used.
int num_threads();

// Corresponds to omp_get_thread_num().
//
// Returns 0 when OpenMP is not used.
int thread_num();

#ifdef _OPENMP

#include <omp.h>

int num_threads() { return omp_get_max_threads(); }

int thread_num() { return omp_get_thread_num(); }

#else

int num_threads() { return 1; }

int thread_num() { return 0; }

#endif

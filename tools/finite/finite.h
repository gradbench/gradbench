// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/finite.h

// Changes made: elimination of trivial warnings.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

const double FINITE_DIFFERENCES_DEFAULT_EPSILON =
    std::cbrt(std::numeric_limits<double>::epsilon());
const double FINITE_DIFFERENCES_DEFAULT_ZERO_NEIGHBORHOOD_RADIUS = 1;
const double FINITE_DIFFERENCES_DEFAULT_ZERO_NEIGHBORHOOD_CHARACTERISTIC_SCALE =
    1;

/// <summary>Generates a function that returns a value proportional to some
/// characteristic scale
///     when the argument lies within a neighborhood of zero or a value
///     proportional to the argument itself otherwise.</summary>
/// <param name="epsilon">Coefficient by which the value of the input variable
/// is multiplied
///     to determine the difference to use in finite differentiation when the
///     absolute value of the said variable is greater or equal to
///     zero_neighborhood_radius</param>
/// <param name="zero_neighborhood_radius">Radius of the neighborhood of zero
/// where the difference
///     used in finite differentiation should not be proportional to the
///     input</param>
/// <param name="zero_neighborhood_characteristic_scale">Value, which when
/// multiplied by epsilon
///     gives the difference to use in finite differentiation when the absolute
///     value of the said variable is lesser than
///     zero_neighborhood_radius</param>
template <typename T>
std::function<T(T)> finite_differences_default_step_function(
    T epsilon = FINITE_DIFFERENCES_DEFAULT_EPSILON,
    T zero_neighborhood_radius =
        FINITE_DIFFERENCES_DEFAULT_ZERO_NEIGHBORHOOD_RADIUS,
    T zero_neighborhood_characteristic_scale =
        FINITE_DIFFERENCES_DEFAULT_ZERO_NEIGHBORHOOD_CHARACTERISTIC_SCALE) {
  return [epsilon, zero_neighborhood_radius,
          zero_neighborhood_characteristic_scale](T x) {
    T absx = std::abs(x);
    return absx >= zero_neighborhood_radius
               ? absx * epsilon
               : zero_neighborhood_characteristic_scale * epsilon;
  };
}

// Engine for arrpoximate differentiation using finite differences.
// Shares temporary memory buffer between calls.
// Encapsulates a function that chooses the step for finite differences
// based on the input value.
template <typename T>
class FiniteDifferencesEngine {
private:
  std::vector<T> tmp_output_f;
  std::vector<T> tmp_output_b;
  int            max_output_size;

  // VECTOR UTILS

  // Add B to A
  static std::vector<T>& add_vec(std::vector<T>& a, const std::vector<T>& b) {
    int sz = a.size();
    for (int i = 0; i < sz; ++i)
      a[i] += b[i];
    return a;
  }

  // Scale vector A by scalar B
  static std::vector<T>& scale_vec(std::vector<T>& a, T b) {
    int sz = a.size();
    for (int i = 0; i < sz; ++i)
      a[i] *= b;
    return a;
  }

  // Insert B starting at a point in A
  static T* vec_ins(T* a, const T* b, int b_sz) {
    for (int i = 0; i < b_sz; ++i)
      a[i] = b[i];
    return a;
  }

  // Compute binomial coefficient.
  static long long binom(int n, int k) {
    if (k > n)
      return 0;
    if (k == 0 || k == n)
      return 1;

    k                = std::min(k, n - k);
    long long result = 1;

    for (int i = 0; i < k; ++i) {
      result *= (n - i);
      result /= (i + 1);
    }

    return result;
  }

public:
  // max_output_size - maximum size of the ouputs of the functions
  // this engine will be able to approximately differentiate.
  // step_function - a function that chooses the step for finite
  // differences based on the input value. Defaults to
  // finite_differences_default_step_function(). For fixed step use
  // [](T x){ return STEP_VALUE; }
  FiniteDifferencesEngine(int                 max_output_size = 0,
                          std::function<T(T)> step_function =
                              finite_differences_default_step_function<T>())
      : tmp_output_f(max_output_size), tmp_output_b(max_output_size),
        max_output_size(max_output_size), step(step_function) {}

  // Computes finite differences step for given argument.
  std::function<T(T)> step;

  // sets max_output_size - maximum size of the ouputs of the functions
  // this engine is be able to approximately differentiate
  void set_max_output_size(int size) {
    tmp_output_f.resize(size);
    tmp_output_b.resize(size);
    max_output_size = size;
  }

  // gets max_output_size - maximum size of the ouputs of the functions
  // this engine is be able to approximately differentiate
  int current_max_output_size() { return max_output_size; }

  // FINITE DIFFERENTIATION FUNCTION

  /// <summary>Approximately differentiate a function using finite differences
  ///     Step for differentiation is determined separately for every input
  ///     by calling this.step(input)</summary>
  ///
  /// <param name="order">The order of the derivative to compute.</param>
  ///
  /// <param name="func">Function to be differentiated.
  ///		Should accept 2 pointers as arguments (one input, one output).
  ///		Each output will be differentiated with respect to each
  ///		input.</param>
  /// <param name="input">Pointer to input data (scalar or vector)</param>
  /// <param name="input_size">Input data size (1 for scalar)</param>
  /// <param name="output_size">Size of 'func' output data</param>
  /// <param name="result">Pointer to where resultant Jacobian should go.
  ///		Will be stored as a vector(input_size * output_size).
  ///		Will store in format foreach (input) { foreach (output) {}
  ///}</param>
  void finite_differences(int order, std::function<void(T*, T*)> func, T* input,
                          int input_size, int output_size, T* result) {
    for (int i = 0; i < input_size; i++) {
      T input_i_orig = input[i];
      T delta        = step(input_i_orig);
      T tmp_b        = input_i_orig - delta;
      T dx           = delta * 2;
      T tmp_f        = tmp_b + dx;
      // adjusting dx so that (tmp_b + dx) - tmp_b == dx
      dx = tmp_f - tmp_b;
      if (dx < delta * 0.5)
        std::cerr << "WARNING: Finite difference step " << delta
                  << " seems incompatible with the argument " << input_i_orig
                  << std::endl;

      std::fill(tmp_output_f.begin(), tmp_output_f.end(), 0);
      for (int j = 0; j < order + 1; j++) {
        input[i] = input_i_orig + (order / 2.0 - j) * dx;
        func(input, tmp_output_b.data());
        scale_vec(tmp_output_b, binom(order, j) * (j % 2 == 0 ? 1 : -1));
        add_vec(tmp_output_f, tmp_output_b);
      }
      input[i] = input_i_orig;

      scale_vec(tmp_output_f, 1 / pow(dx, order));
      vec_ins(&result[output_size * i], tmp_output_f.data(), output_size);
    }
  }
};

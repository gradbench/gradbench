// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualLSTM.cpp

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include "lstm_d.hpp"

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
public:
  Jacobian(lstm::Input& input) : Function(input) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    double err;
    lstm_objective_d(_input.l, _input.c, _input.b,
                     _input.main_params.data(), _input.extra_params.data(),
                     _input.state, _input.sequence.data(),
                     &err, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}

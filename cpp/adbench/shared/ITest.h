// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/ITest.h
//
// Changes made:
//
//   - removed 'times' parameter from calculate_objective/calculate_jacobian.
//   - changed 'prepare' method into a constructor.
//   - added a 'prepare_jacobian' method.
//   - added _input/_output fields.

#pragma once

template <typename Input, typename Output>
class ITest {
protected:
  Input& _input;
  Output _output;
public:
  // This function must be called before any other function.
  ITest(Input& input) : _input(input) {}
  // This function must be called before calculate_jacobian.
  virtual void prepare_jacobian() { };
  // calculate function
  virtual void calculate_objective() = 0;
  virtual void calculate_jacobian() = 0;
  Output output() { return _output; };
  virtual ~ITest() = default;
};

// Factory function that creates instances of the GMMTester object.
// Should be declared in each module.


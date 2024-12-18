#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class AdolCGMM : public ITest<GMMInput, GMMOutput> {
  GMMInput _input;
  GMMOutput _output;

public:
  void prepare(GMMInput&& input) override;

  void calculate_objective(int times) override;
  void calculate_jacobian(int times) override;
  GMMOutput output() override;

  ~AdolCGMM() = default;
};

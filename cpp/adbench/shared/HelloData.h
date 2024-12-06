#pragma once

#include <vector>

#include "defs.h"
#include "utils.h"

struct HelloInput {
  double x;
};

struct HelloOutput {
  double objective;
  double gradient;
};

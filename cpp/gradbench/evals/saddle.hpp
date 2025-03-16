#pragma once

#include "json.hpp"

namespace saddle {
// Will always be of length 2.
typedef std::vector<double> Input;

// Will always be of length 4.
typedef std::vector<double> Output;

// The underlying function that we try to find saddle points for.
template<typename T>
T objective(T x1, T y1, T x2, T y2) {
  return (x1*x1 + y1*y1) - (x2*x2 + y2*y2);
}

}

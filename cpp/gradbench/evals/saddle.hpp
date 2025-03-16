#pragma once

#include "json.hpp"

namespace saddle {
struct Input {
  double start[2];
};

// Will always be of length 4.
typedef std::vector<double> Output;

// The underlying function that we try to find saddle points for.
template<typename T>
T objective(T x1, T y1, T x2, T y2) {
  return (x1*x1 + y1*y1) - (x2*x2 + y2*y2);
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  std::vector<double> start = j.at("start");
  p.start[0] = start.at(0);
  p.start[1] = start.at(1);
}

}

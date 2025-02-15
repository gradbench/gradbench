#pragma once

#include "gradbench/main.hpp"

namespace hello {

template<typename D>
D square(D x) {
  return x * x;
}

typedef double Input;
typedef double SquareOutput;
typedef double DoubleOutput;

class Square : public Function<Input, SquareOutput> {
public:
  Square(Input& input) : Function(input) {}

  void compute(SquareOutput& output) {
    output = _input * _input;
  }
};

}

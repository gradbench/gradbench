#include "gradbench/evals/hello.hpp"
#include "adept.h"
#include "gradbench/main.hpp"
#include <iostream>

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    using adept::adouble;
    using adept::Real;
    adept::Stack s;
    adouble      x = _input;

    s.new_recording();
    adouble y = hello::square<adouble>(x);
    y.set_gradient(1.0);
    s.reverse();
    s.independent(&x, 1);
    s.dependent(y);
    s.jacobian(&output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
  ;
}

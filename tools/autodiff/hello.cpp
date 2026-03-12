#include "gradbench/evals/hello.hpp"
#include "gradbench/main.hpp"
#include <iostream>

#include <autodiff/forward/dual.hpp>
using namespace autodiff;

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {
  }

  void compute(hello::DoubleOutput& output) {
    dual x =_input;
    output = derivative(hello::square<dual>, wrt(x), at(x));
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
  ;
}

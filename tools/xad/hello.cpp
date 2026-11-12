#include "gradbench/evals/hello.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>

class Double : public Function<hello::Input, hello::DoubleOutput> {
  using AD = xad::fwd<double>::active_type;

public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    AD x               = _input;
    xad::derivative(x) = 1.0;
    AD y               = hello::square<AD>(x);
    output             = xad::derivative(y);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
}

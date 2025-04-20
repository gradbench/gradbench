#include "gradbench/evals/hello.hpp"
#include "gradbench/main.hpp"
#include <codi.hpp>

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    codi::RealForward x(_input);
    x.setGradient(1);
    auto y = hello::square<codi::RealForward>(x);
    output = y.gradient();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
  ;
}

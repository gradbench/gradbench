#include "gradbench/evals/hello.hpp"
#include "gradbench/main.hpp"

extern double __enzyme_autodiff(void*, double);

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    output = __enzyme_autodiff((void*)hello::square<double>, _input);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
  ;
}

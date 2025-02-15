#include "gradbench/main.hpp"
#include "gradbench/evals/hello.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Double : public Function<hello::Input, hello::DoubleOutput> {
  CppAD::ADFun<double> *_tape;

public:
  Double(hello::Input& input) : Function(input) {
    std::vector<ADdouble> X(1);
    std::vector<ADdouble> Y(1);
    X[0] = _input;
    CppAD::Independent(X);

    Y[0] = hello::square(X[0]);
    _tape = new CppAD::ADFun<double>(X, Y);
  }

  void compute(hello::DoubleOutput& output) {
    std::vector<double> dx(1);
    dx[0] = 1;
    std::vector<double> dy(1);

    dy = _tape->Forward(1, dx);
    output = dy[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"square", function_main<hello::Square>},
      {"double", function_main<Double>}
    });;
}

#include <iostream>

#include "AdeptHello.h"
#include "adbench/main.h"
#include "adbench/shared/HelloData.h"

int main(int argc, char* argv[]) {
  return generic_main<HelloInput,
                      AdeptHello,
                      read_HelloInput_json,
                      write_HelloOutput_objective_json,
                      write_HelloOutput_jacobian_json>(argc, argv);
}

/*
#include "adept.h"
#include "adbench/shared/hello.h"


int main(int argc, char** argv) {
  using adept::adouble;
  using adept::Real;

  adept::Stack s;

  adouble x = 5, y;

  s.new_recording();
  y = hello_objective(x);
  y.set_gradient(1.0);
  s.reverse();
  s.independent(&x, 1);
  s.dependent(y);
  Real J[1];
  s.jacobian(J);
  std::cout << J[0] << std::endl;
}

*/

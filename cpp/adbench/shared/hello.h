#pragma once

// Written as a template function to allow for overloading-based AD
// tools.
template<typename D>
D hello_objective(D x) {
  return x * x;
}

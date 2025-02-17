// A generic main function for tools built on C++.

#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include "json.hpp"

// To implement a function for an eval, define a class that inherits
// from this one. The InputP type must be deserialisable from JSON
// (with the to_json overloaded function), and OutputP must be
// serialisable to JSON (with from_json).
template <typename InputP, typename OutputP>
class Function {
protected:
  InputP& _input;
public:
  typedef InputP Input;
  typedef OutputP Output;
  Function(Input& input) : _input(input) {}
  virtual ~Function() = default;
  virtual void compute(Output&) = 0;
};

template<typename Benchmark>
int function_main(const std::string& input_file) {
  using namespace std::chrono;
  using json = nlohmann::json;
  std::ifstream f(input_file);
  json j = json::parse(f);
  typename Benchmark::Input input = j.template get<typename Benchmark::Input>();

  int runs = j.is_object() ? int(j["runs"]) : 1;
  assert(runs > 0);

  auto prepare_start = high_resolution_clock::now();
  Benchmark b(input);
  auto prepare_finish = high_resolution_clock::now();
  long prepare_time_taken = duration_cast<nanoseconds>(prepare_finish - prepare_start).count();

  std::vector<high_resolution_clock::time_point> start(runs), finish(runs);

  typename Benchmark::Output output;

  for (int i = 0; i < runs; i++) {
    start[i] = high_resolution_clock::now();
    b.compute(output);
    finish[i] = high_resolution_clock::now();
  }

  std::cout << json(output) << std::endl;

  std::cout << "{\"name\": \"prepare\", \"nanoseconds\": " << (long)prepare_time_taken << "}" << std::endl;

  for (int i = 0; i < runs; i++) {
    long time_taken = duration_cast<nanoseconds>(finish[i] - start[i]).count();
    std::cout << "{\"name\": \"evaluate\", \"nanoseconds\": " << (long)time_taken << "}" << std::endl;
  }

  return 0;
}

int generic_main(int argc, char* argv[],
                 std::map<std::string, int (*)(const std::string&)> &&m) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " FILE FUNCTION" << std::endl;
    exit(1);
  }

  std::string input_file = argv[1];
  std::string function = argv[2];

  for (auto it : m) {
    if (it.first == function) {
      return it.second(input_file);
    }
  }

  std::cerr << "Unknown function: " << function << std::endl;
  return 1;
}

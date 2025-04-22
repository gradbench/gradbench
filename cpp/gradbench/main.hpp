// A generic main function for tools built on C++.

#pragma once

#include "json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

// To implement a function for an eval, define a class that inherits
// from this one. The InputP type must be deserialisable from JSON
// (with the to_json overloaded function), and OutputP must be
// serialisable to JSON (with from_json).
template <typename InputP, typename OutputP>
class Function {
protected:
  InputP& _input;

public:
  typedef InputP  Input;
  typedef OutputP Output;
  Function(Input& input) : _input(input) {}
  virtual ~Function()           = default;
  virtual void compute(Output&) = 0;
};

template <typename Benchmark>
int function_main(const std::string& input_file) {
  using namespace std::chrono;
  using json = nlohmann::json;
  std::ifstream             f(input_file);
  json                      j     = json::parse(f);
  typename Benchmark::Input input = j.template get<typename Benchmark::Input>();

  int    min_runs    = j.contains("min_runs") ? int(j["min_runs"]) : 1;
  double min_seconds = j.contains("min_seconds") ? double(j["min_seconds"]) : 0;
  assert(min_runs > 0);

  auto      prepare_start = steady_clock::now();
  Benchmark b(input);
  auto      prepare_finish = steady_clock::now();
  long      prepare_time_taken =
      duration_cast<nanoseconds>(prepare_finish - prepare_start).count();

  std::vector<long> evaluate_times;

  typename Benchmark::Output output;
  double                     elapsed_seconds = 0;
  ;

  for (int i = 0; i < min_runs || elapsed_seconds < min_seconds; i++) {
    auto start = steady_clock::now();
    b.compute(output);
    auto finish     = steady_clock::now();
    long elapsed_ns = duration_cast<nanoseconds>(finish - start).count();
    evaluate_times.push_back(elapsed_ns);
    elapsed_seconds += (double)elapsed_ns / 1e9;
  }

  std::cout << json(output) << std::endl;

  std::cout << "{\"name\": \"prepare\", \"nanoseconds\": "
            << (long)prepare_time_taken << "}" << std::endl;

  for (auto t : evaluate_times) {
    std::cout << "{\"name\": \"evaluate\", \"nanoseconds\": " << t << "}"
              << std::endl;
  }

  return 0;
}

int generic_main(int argc, char* argv[],
                 std::map<std::string, int (*)(const std::string&)>&& m) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " FILE FUNCTION" << std::endl;
    exit(1);
  }

  std::string input_file = argv[1];
  std::string function   = argv[2];

  for (auto it : m) {
    if (it.first == function) {
      return it.second(input_file);
    }
  }

  std::cerr << "Unknown function: " << function << std::endl;
  return 1;
}

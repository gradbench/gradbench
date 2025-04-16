// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Based on:
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/lstm.h
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/LSTMData.h

#pragma once

#include <cmath>
#include <vector>

#include "gradbench/main.hpp"
#include "json.hpp"

namespace lstm {
// UTILS

// Sigmoid on scalar
template <typename T>
T sigmoid(T x) {
  return 1 / (1 + exp(-x));
}

// log(sum(exp(x), 2))
template <typename T>
T logsumexp(const T* vect, int sz) {
  T sum = 0.0;
  for (int i = 0; i < sz; ++i)
    sum += exp(vect[i]);
  sum += 2;
  return log(sum);
}

// log(sum(exp(x), 2))
template <typename T>
T logsumexp_store_temps(const T* vect, int sz) {
  std::vector<T> vect2(sz);
  for (int i = 0; i < sz; ++i)
    vect2[i] = exp(vect[i]);
  T sum = 0.0;
  for (int i = 0; i < sz; ++i)
    sum += vect2[i];
  sum += 2;
  return log(sum);
}

// Helper structures

template <typename T>
struct WeightOrBias {
  const T* forget;
  const T* ingate;
  const T* outgate;
  const T* change;

  WeightOrBias(const T* params, int hsize)
      : forget(params), ingate(&params[hsize]), outgate(&params[2 * hsize]),
        change(&params[3 * hsize]) {}
};

template <typename T>
struct LayerParams {
  const WeightOrBias<T> weight;
  const WeightOrBias<T> bias;

  LayerParams(const T* layer_params, int hsize)
      : weight(layer_params, hsize), bias(&layer_params[hsize * 4], hsize) {}
};

template <typename T>
struct MainParams {
  std::vector<LayerParams<T>> layer_params;

  MainParams(const T* main_params, int hsize, int n_layers) {
    layer_params.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
      layer_params.emplace_back(&main_params[8 * hsize * i], hsize);
    }
  }
};

template <typename T>
struct ExtraParams {
  const T* in_weight;
  const T* out_weight;
  const T* out_bias;

  ExtraParams(const T* params, int hsize)
      : in_weight(params), out_weight(&params[hsize]),
        out_bias(&params[2 * hsize]) {}
};

template <typename T>
struct InputSequence {
  std::vector<const T*> sequence;

  InputSequence(const T* input_sequence, int char_bits, int char_count) {
    sequence.reserve(char_count);
    for (int i = 0; i < char_count; ++i) {
      sequence.push_back(&input_sequence[char_bits * i]);
    }
  }
};

template <typename T>
struct LayerState {
  T* hidden;
  T* cell;

  LayerState(T* layer_state, int hsize)
      : hidden(layer_state), cell(&layer_state[hsize]) {}
};

template <typename T>
struct State {
  std::vector<LayerState<T>> layer_state;

  State(T* state, int hsize, int n_layers) {
    layer_state.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
      layer_state.emplace_back(&state[2 * hsize * i], hsize);
    }
  }
};

// LSTM OBJECTIVE

// The LSTM model
template <typename T>
void lstm_model(int hsize, const LayerParams<T>& params, LayerState<T>& state,
                const T* input) {
  for (int i = 0; i < hsize; ++i) {
    // gates for i-th cell/hidden
    T forget =
        sigmoid<T>(input[i] * params.weight.forget[i] + params.bias.forget[i]);
    T ingate  = sigmoid<T>(state.hidden[i] * params.weight.ingate[i] +
                           params.bias.ingate[i]);
    T outgate = sigmoid<T>(input[i] * params.weight.outgate[i] +
                           params.bias.outgate[i]);
    T change =
        tanh(state.hidden[i] * params.weight.change[i] + params.bias.change[i]);

    state.cell[i]   = state.cell[i] * forget + ingate * change;
    state.hidden[i] = outgate * tanh(state.cell[i]);
  }
}

// Predict LSTM output given an input
template <typename T>
void lstm_predict(int l, int b, const MainParams<T>& main_params,
                  const ExtraParams<T>& extra_params, State<T>& state,
                  const T* input, T* output) {
  for (int i = 0; i < b; ++i)
    output[i] = input[i] * extra_params.in_weight[i];

  T* layer_output = output;

  for (int i = 0; i < l; ++i) {
    lstm_model(b, main_params.layer_params[i], state.layer_state[i],
               layer_output);
    layer_output = state.layer_state[i].hidden;
  }

  for (int i = 0; i < b; ++i)
    output[i] =
        layer_output[i] * extra_params.out_weight[i] + extra_params.out_bias[i];
}

// LSTM objective (loss function)
template <typename T>
void objective(int l, int c, int b, const T* main_params, const T* extra_params,
               const T* state, const T* sequence, T* loss) {
  T              total = 0.0;
  int            count = 0;
  MainParams<T>  main_params_wrap(main_params, b, l);
  ExtraParams<T> extra_params_wrap(extra_params, b);
  std::vector<T> state_copy(l * 2 * b);
  std::copy(&state[0], &state[l * 2 * b], state_copy.data());
  State<T>         state_wrap(state_copy.data(), b, l);
  InputSequence<T> sequence_wrap(sequence, b, c);
  std::vector<T>   ypred(b), ynorm(b);
  for (int t = 0; t < c - 1; ++t) {
    lstm_predict(l, b, main_params_wrap, extra_params_wrap, state_wrap,
                 sequence_wrap.sequence[t], ypred.data());

    T lse = logsumexp(ypred.data(), b);
    for (int i = 0; i < b; ++i)
      ynorm[i] = ypred[i] - lse;

    const T* ygold = sequence_wrap.sequence[t + 1];
    for (int i = 0; i < b; ++i)
      total += ygold[i] * ynorm[i];

    count += b;
  }

  *loss = -total / count;
}

struct Input {
  int                 l;
  int                 c;
  int                 b;
  std::vector<double> main_params;
  std::vector<double> extra_params;
  std::vector<double> state;
  std::vector<double> sequence;
};

typedef double ObjOutput;

typedef std::vector<double> JacOutput;

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  auto main_params  = j["main_params"].get<std::vector<std::vector<double>>>();
  auto extra_params = j["extra_params"].get<std::vector<std::vector<double>>>();
  auto state        = j["state"].get<std::vector<std::vector<double>>>();
  auto sequence     = j["sequence"].get<std::vector<std::vector<double>>>();

  p.l = main_params.size() / 2;
  p.b = main_params[0].size() / 4;
  p.c = sequence.size();

  for (auto it = main_params.begin(); it != main_params.end(); it++) {
    p.main_params.insert(p.main_params.end(), it->begin(), it->end());
  }

  for (auto it = extra_params.begin(); it != extra_params.end(); it++) {
    p.extra_params.insert(p.extra_params.end(), it->begin(), it->end());
  }

  for (auto it = state.begin(); it != state.end(); it++) {
    p.state.insert(p.state.end(), it->begin(), it->end());
  }

  for (auto it = sequence.begin(); it != sequence.end(); it++) {
    p.sequence.insert(p.sequence.end(), it->begin(), it->end());
  }
}

class Objective : public Function<Input, ObjOutput> {
public:
  Objective(Input& input) : Function(input) {}

  void compute(ObjOutput& output) {
    objective(_input.l, _input.c, _input.b, _input.main_params.data(),
              _input.extra_params.data(), _input.state.data(),
              _input.sequence.data(), &output);
  }
};
}  // namespace lstm

#include "nn.hpp"
#include "value.h"
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>

Neuron::Neuron(size_t inputSize) {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1, 1);

  w.reserve(inputSize);
  for (size_t i = 0; i < inputSize; ++i)
    w.push_back(std::make_shared<Value>(dist(rng)));

  b = std::make_shared<Value>(dist(rng));
}

std::shared_ptr<Value>
Neuron::operator()(std::vector<std::shared_ptr<Value>> &input) {
  if (input.size() != w.size())
    throw std::runtime_error("Input and weight size mismatch");

  std::shared_ptr<Value> out = *w[0] * input[0];
  for (size_t i = 1; i < input.size(); ++i)
    out = *out + (*w[i] * input[i]);

  out = *out + b;
  return out->tanh();
}

// #include "nn.hpp"
// #include "value.h"
// #include <cstddef>
// #include <random>
// #include <stdexcept>
//
// Neuron::Neuron(size_t inputSize) : w(inputSize) {
//   static std::mt19937 rng(std::random_device{}());
//   std::uniform_real_distribution<double> dist(-1, 1);
//   for (Value &weight : w)
//     weight.setData(dist(rng));
//   b = dist(rng);
// }
//
// Value Neuron::operator()(std::vector<Value> &input) {
//   if (input.size() != w.size())
//     throw std::runtime_error("Input and weight size mismatch");
//
//   Value out(0.0);
//   for (size_t i = 0; i < input.size(); ++i)
//     out = out + (w[i] * input[i]);
//
//   out = out + b;
//   return out.tanh();
// }

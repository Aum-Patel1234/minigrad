#include "value.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(minigrad, m) {
  m.doc() = "MiniGrad Value class Python bindings";

  py::class_<Value>(m, "Value")
      .def(py::init<double>(), "Constructor with a double value")
      .def("getData", &Value::getData, "Get the internal data")
      // Overload + operator
      .def(py::self + py::self)
      .def(py::self * py::self)
      .def(py::self - py::self)
      .def(py::self / py::self)
      .def("tanh", &Value::tanh)
      .def("getLabel", &Value::getLabel)
      .def("getOp", &Value::getOp)
      .def("getGrad", &Value::getGrad)
      // Backpropagation
      .def("backPropogate", &Value::backPropogate,
           "Perform backward pass to compute gradients")
      // Optional: define __repr__ for nice printing in Python
      .def("__repr__",
           [](const Value &v) {
             std::string label =
                 std::string(v.getLabel()); // convert string_view -> string
             if (label.empty()) {
               label = "Value(" + std::to_string(v.getData()) + ")";
             } else {
               label += "\\nValue(" + std::to_string(v.getData()) + ")";
             }
             return "<" + label + ">";
           })
      .def("exportGraph", [](const Value &v) {
        py::dict result;

        std::vector<const Value *> nodes;
        std::vector<std::pair<const Value *, const Value *>> edges;
        v.buildGraph(nodes, edges);

        result["nodes"] = nodes;
        result["edges"] = edges;

        return result;
      });
}

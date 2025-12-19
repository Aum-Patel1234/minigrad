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
      // Optional: define __repr__ for nice printing in Python
      .def("__repr__", [](const Value &v) {
        return "<Value data=" + std::to_string(v.getData()) + ">";
      });
}

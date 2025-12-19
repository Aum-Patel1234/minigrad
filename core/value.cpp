#include "value.h"

Value::Value(double data) : data(data) {}

double Value::getData() const { return this->data; }

Value Value::operator+(const Value &other) const {
  return Value(this->data + other.data);
}

Value Value::operator-(const Value &other) const {
  return Value(this->data - other.data);
}

Value Value::operator*(const Value &other) const {
  return Value(this->data * other.data);
}

Value Value::operator/(const Value &other) const {
  return Value(this->data / other.data);
}

std::ostream &operator<<(std::ostream &os, const Value &val) {
  os << "Value(data=" << val.data << ")\n";
  return os;
}

#pragma once

#include "gradbench/main.hpp"
#include "json.hpp"

namespace particle {
struct Input {
  double w0;
};

typedef double Output;

template <typename T>
class Point {
public:
  T x, y;
  Point() : x(0), y(0) {}
  Point(T px, T py) : x(px), y(py) {}
};

template <typename T>
Point<T> operator+(const T& k, const Point<T>& p) {
  return Point(k + p.x, k + p.y);
}

template <typename T>
Point<T> operator*(const T& k, const Point<T>& p) {
  return Point(k * p.x, k * p.y);
}

template <typename T>
Point<T> operator*(const Point<T>& u, const Point<T>& v) {
  return Point(u.x * v.x, u.y * v.y);
}

template <typename T>
Point<T> operator+(const Point<T>& u, const Point<T>& v) {
  return Point(u.x + v.x, u.y + v.y);
}

template <typename T>
Point<T> operator+=(Point<T>& u, const Point<T>& v) {
  u.x += v.x;
  u.y += v.y;
  return u;
}

template <typename T>
double dist(const Point<T>& u, const Point<T>& v) {
  return sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y));
}

template <typename T>
T accel(const std::vector<Point<T>>& charges, const Point<T>& x) {
  T a = 0;
  for (auto p : charges) {
    a += 1 / dist(p, x);
  }
  return a;
}

using json = nlohmann::json;

void from_json(const json& j, Input& p) { p.w0 = j.at("w"); }
}  // namespace particle

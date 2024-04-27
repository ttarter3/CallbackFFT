#ifndef CUMATH_CUH
#define CUMATH_CUH

#include <complex>
#include <cuComplex.h>


__device__ cuComplex Div(cuComplex a, cuComplex b) {
  cuComplex c;
  float denominator = b.x * b.x + b.y * b.y;
  c.x = (a.x * b.x + a.y * b.y) / denominator;
  c.y = (a.y * b.x - a.x * b.y) / denominator;
  return c;
}

__device__ cuComplex Mult(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__device__ __inline__ cuComplex Add(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__device__ cuComplex XAdd(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x + b.y;
  c.y = a.y + b.x;
  return c;
}

__device__ __inline__ cuComplex Add(cuComplex a, cuComplex b,cuComplex c, cuComplex d) {
  cuComplex e;
  e.x = a.x + b.x + c.x + d.x;
  e.y = a.y + b.y + c.y + d.y;
  return e;
}

__device__ cuComplex Sub(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

__device__ cuComplex XSub(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x - b.y;
  c.y = a.y - b.x;
  return c;
}

__device__ cuComplex XAddSub(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x + b.y;
  c.y = a.y - b.x;
  return c;
}

__device__ cuComplex XSubAdd(cuComplex a, cuComplex b) {
  cuComplex c;
  c.x = a.x - b.y;
  c.y = a.y + b.x;
  return c;
}

__device__ cuComplex Sub(cuComplex a, cuComplex b,cuComplex c, cuComplex d) {
  cuComplex e;
  e.x = a.x - b.x - c.x - d.x;
  e.y = a.y - b.y - c.x - d.x;
  return e;
}
__device__ void Swap(cuComplex &a, cuComplex &b) {
  cuComplex temp = a;
  a = b;
  b = temp;
}

#endif
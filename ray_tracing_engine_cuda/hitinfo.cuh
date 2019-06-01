#ifndef HITINFO_CUDA_H

#include "ray.cuh"
class hitinfo {
public:
  float dis;
  vector3 pos;
  vector3 normal;

  Hitinfo() = default;
}
#define HITINFO_CUDA_H
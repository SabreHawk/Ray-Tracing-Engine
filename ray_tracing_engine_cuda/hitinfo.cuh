#ifndef HITINFO_CUDA_H
#define HITINFO_CUDA_H
#include "ray.cuh"
class hitinfo {
public:
  float dis;
  vector3 pos;
  vector3 normal;

  __host__ __device__ hitinfo() {}
};
#endif
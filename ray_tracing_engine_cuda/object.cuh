//
// Created by Zhiquan on 2019/6/1.
//

#ifndef RAY_TRACING_ENGINE_OBJECT_CUDA_H
#define RAY_TRACING_ENGINE_OBJECT_CUDA_H

#include "hitinfo.cuh"
#include "ray.cuh"

class Object {
protected:
  Material *material_ptr;

public:
  __device__ virtual bool hit(const Ray &, double, double, HitInfo &) const = 0;

  virtual void dispInfo() = 0;
};

#endif // RAY_TRACING_ENGINE_OBJECT_CUDA_H

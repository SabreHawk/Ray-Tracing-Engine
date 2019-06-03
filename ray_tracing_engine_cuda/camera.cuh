//
// Created by mySab on 2019/06/02.
//

#ifndef RAY_TRACING_ENGINE_CAMERA_CUDA_H
#define RAY_TRACING_ENGINE_CAMERA_CUDA_H
#include "ray.cuh"

class camera {
public:
  vector3 origin;
  vector3 lower_left_corner;
  vector3 horizontal_vec;
  vector3 vertical_vec;
  __device__ camera(const vector3 &, const vector3 &, const vector3 &,
                    const vector3 &);
  __device__ ray gen_ray(const float &, const float &);
};

__device__ camera::camera(const vector3 &_origin,
                          const vector3 &_lower_left_corner,
                          const vector3 &_horizontal_vec,
                          const vector3 &_vertical_vec) {
  origin = _origin;
  lower_left_corner = _lower_left_corner;
  horizontal_vec = _horizontal_vec;
  vertical_vec = _vertical_vec;
}

__device__ ray camera::gen_ray(const float &_u, const float &_v) {
  return ray(origin, lower_left_corner + _u * horizontal_vec +
                         _v * vertical_vec - origin);
}

// class camera {
// public:
//   __device__ camera() {
//     lower_left_corner = vector3(-2.0, -1.0, -1.0);
//     horizontal = vector3(4.0, 0.0, 0.0);
//     vertical = vector3(0.0, 2.0, 0.0);
//     origin = vector3(0.0, 0.0, 0.0);
//   }
//   __device__ ray gen_ray(float u, float v) {
//     return ray(origin,
//                lower_left_corner + u * horizontal + v * vertical - origin);
//   }

//   vector3 origin;
//   vector3 lower_left_corner;
//   vector3 horizontal;
//   vector3 vertical;
// };

#endif // RAY_TRACING_ENGINE_CAMERA_H
//
// Created by mySab on 2018/9/22.
//

#ifndef RAY_TRACING_ENGINE_SPHERE_H
#define RAY_TRACING_ENGINE_SPHERE_H

#include "Object.h"

class sphere : public object {
private:
  vector3 center;
  float radius;

public:
  sphere() = default;

  __device__ sphere::sphere(const vector3 &_c, const float &_r);

  __device__ bool hit(const Ray &, double, double, HitInfo &) const override;

  bool displacement(const double &_time, Vector3 &_target_pos) const override;

  void dispInfo() override;
};
__device__ sphere::sphere(const vector3 &_c, const float &_r)
    : center(_c), radius(_r) {}
__device__ bool sphere::hit(const ray &r, float d_min, float d_max,
                            hit_record &rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float tmp = (-b - sqrt(discriminant)) / a;
    if (d_min < tmp && tmp < d_max) {
      rec.dis = tmp;
      rec.p = r.point_at_parameter(rec.dis);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
    tmp = (-b + sqrt(discriminant)) / a;
    if (d_min < tmp && tmp < d_max) {
      rec.dis = tmp;
      rec.p = r.target_pos(rec.dis);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
  }
  return false;
}
#endif // RAY_TRACING_ENGINE_SPHERE_H

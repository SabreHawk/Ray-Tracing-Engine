//
// Created by mySab on 2018/9/27.
//

#include "Metal.h"
#include "HitInfo.h"

Metal::Metal(const Vector3 &_a, const double &_f) : albedo(_a), fuzz(_f > 1.0 ? 1.0 : _f) {
}

bool Metal::scatter(const Ray &_in_ray, const HitInfo &_hit_info, Vector3 &_attenuation, Ray &_scattered_ray) const {
    Vector3 ref_vec = reflect(_in_ray.direction().normalize(), _hit_info.normal);
    _scattered_ray = Ray(_hit_info.pos, ref_vec + fuzz * random_in_unit_sphere(), _in_ray.get_time());
    _attenuation = albedo;
    return (dot(_scattered_ray.direction(), _hit_info.normal) > 0);
}

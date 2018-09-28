//
// Created by mySab on 2018/9/27.
//

#include "Metal.h"
#include "HitInfo.h"

Metal::Metal(const Vector3 &_a) : albedo(_a) {

}

bool Metal::scatter(const Ray &_r, const HitInfo &_i, Vector3 &_a, Ray &_s) const {
    Vector3 ref_vec = reflect(_r.direction().normalize(), _i.normal);
    _s = Ray(_i.pos, ref_vec);
    _a = albedo;
    return (dot(_s.direction(), _i.normal) > 0);
}

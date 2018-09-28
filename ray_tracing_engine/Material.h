//
// Created by mySab on 2018/9/26.
//

#ifndef RAY_TRACING_ENGINE_MATERIAL_H
#define RAY_TRACING_ENGINE_MATERIAL_H


#include "Ray.h"
class HitInfo;
class Material {
public:
    virtual bool scatter(const Ray &,const HitInfo&,Vector3 &,Ray &) const = 0;
};

Vector3 reflect(const Vector3& _v,const Vector3& _n);
Vector3 random_in_unit_sphere();
#endif //RAY_TRACING_ENGINE_MATERIAL_H

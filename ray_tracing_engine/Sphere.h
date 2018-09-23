//
// Created by mySab on 2018/9/22.
//

#ifndef RAY_TRACING_ENGINE_SPHERE_H
#define RAY_TRACING_ENGINE_SPHERE_H

#include "Object.h"

class Sphere : public Object {
private:
    Vector3 center;
    double radius;
public:
    Sphere();

    Sphere(const Vector3 &, double);

    virtual bool hit(const Ray &, double, double, hitInfo &) const;

    virtual void dispInfo();
};


#endif //RAY_TRACING_ENGINE_SPHERE_H

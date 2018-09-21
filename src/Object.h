//
// Created by Zhiquan on 2018/9/20.
//

#ifndef RAY_TRACING_ENGINE_OBJECT_H
#define RAY_TRACING_ENGINE_OBJECT_H

#include "Ray.h"
#include "Vector3.h"

struct hitInfo {
    double t;
    Vector3 pos;
    Vecotr3 normal;
};

class Object {
public:
    Object() = default;
    virtual bool hit(const Ray &, double, double, hitInfo &) const = 0;

};


#endif //RAY_TRACING_ENGINE_OBJECT_H

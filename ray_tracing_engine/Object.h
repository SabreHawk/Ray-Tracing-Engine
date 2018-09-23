//
// Created by Zhiquan on 2018/9/20.
//

#ifndef RAY_TRACING_ENGINE_OBJECT_H
#define RAY_TRACING_ENGINE_OBJECT_H

#include <iostream>
#include "Ray.h"

struct hitInfo {
    double t;
    Vector3 pos;
    Vector3 normal;
};

class Object {
public:
    virtual bool hit(const Ray &, double, double, hitInfo &) const=0;
    virtual void dispInfo() = 0;
};


#endif //RAY_TRACING_ENGINE_OBJECT_H

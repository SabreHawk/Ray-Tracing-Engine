//
// Created by Zhiquan on 2018/9/20.
//

#ifndef RAY_TRACING_ENGINE_OBJECT_H
#define RAY_TRACING_ENGINE_OBJECT_H

#include <iostream>
#include "Ray.h"
#include "Vector3.h"

struct hitInfo {
    double t;
    Vector3 pos;
    Vector3 normal;
};

class Object {
public:
    virtual bool hit(const Ray &, double, double, hitInfo &) const {
        return true;
    }
};


#endif //RAY_TRACING_ENGINE_OBJECT_H

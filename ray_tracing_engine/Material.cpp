//
// Created by mySab on 2018/9/27.
//

#include "Material.h"

Vector3 random_in_unit_sphere() {
    Vector3 p;
    do {
        p = 2.0 * Vector3(drand48(), drand48(), drand48()) - Vector3(1, 1, 1);
    } while (p.length() * p.length() >= 1.0);
    return p;
}

Vector3 reflect(const Vector3& _v,const Vector3& _n){
    return _v - 2 * dot(_v,_n) * _n;
}

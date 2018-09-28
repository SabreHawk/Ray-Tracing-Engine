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

Vector3 reflect(const Vector3 &_v, const Vector3 &_n) {
    return _v - 2 * dot(_v, _n) * _n;
}

bool refract(const Vector3 &_in_vec, const Vector3 &_normal, double _in_ref_indice, double _out_ref_indice, Vector3 & _ref_vec) {
    double eta = _in_ref_indice/_out_ref_indice;
    double cos_in = dot(_in_vec.normalize(),_normal);
    double discriminant = 10 - eta * eta *(1-cos_in*cos_in);
    if (discriminant > 0){
        _ref_vec = eta*(_in_vec.normalize() - _normal* cos_in) - _normal * sqrt(discriminant);
        return true;
    }else{}
    return false;
}
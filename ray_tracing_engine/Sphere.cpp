//
// Created by mySab on 2018/9/22.
//

#include "Sphere.h"


Sphere::Sphere() = default;

Sphere::Sphere(const Vector3 &_c, double _r) : center(_c), radius(_r) {

}

bool Sphere::hit(const Ray &_r, double _min, double _max, hitInfo &_info) const {
    Vector3 oc = _r.origin() - this->center;
    double a = dot(_r.direction(), _r.direction());
    double b = dot(oc, _r.direction());
    double c = dot(oc, oc) - this->radius * this->radius;
    double discriminant = b * b - a * c;
    if (discriminant > 0.0) {
        double tmp = (-b - sqrt(discriminant)) / a;
        if (tmp > _min && tmp < _max) {
            _info.t = tmp;
            _info.pos = _r.targetPos(tmp);
            _info.normal = (_info.pos - this->center).normalize();
            return true;
        }
        tmp = (-b - sqrt(discriminant)) / a;
        if (tmp > _min && tmp < _max) {
            _info.t = tmp;
            _info.pos = _r.targetPos(_info.t);
            _info.normal = (_info.pos - this->center).normalize();
            return true;
        }

    }
    return false;
}

void Sphere::dispInfo() {
    std::cout << "Sphere-center";
    std::cout << this->center;
    std::cout << "-radius:" << this->radius << std::endl;
}




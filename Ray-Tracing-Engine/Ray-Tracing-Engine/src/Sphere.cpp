//
// Created by mySab on 2018/9/22.
//

#include "Sphere.h"

std::ostream &operator<<(std::ostream &_os, const Sphere &_r) {
    std::cout << "Sphere-center";
    std::cout << _r.center;
    std::cout << "-radius:" << _r.radius;
}




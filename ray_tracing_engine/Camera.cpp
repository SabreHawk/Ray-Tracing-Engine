//
// Created by mySab on 2018/9/24.
//

#include "Camera.h"

Camera::Camera() : origin(Vector3(0, 0, 0)),
                   low_left_corner(Vector3(0, 0, 0)),
                   horizontal_vec(Vector3(0, 0, 0)),
                   vertical_vec(Vector3(0, 0, 0)) {
}

Camera::Camera(const Vector3 &_o, const Vector3 &_l, const Vector3 &_h, const Vector3 &_v)
        : origin(_o), low_left_corner(_l), horizontal_vec(_h), vertical_vec(_v) {

}

Ray Camera::gen_ray(double _u, double _v) {
    Ray tmp_r(origin, low_left_corner + _u * horizontal_vec + _v * vertical_vec - origin);
    return {tmp_r};
}

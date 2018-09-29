//
// Created by mySab on 2018/9/24.
//

#include "Camera.h"

Camera::Camera() : origin(Vector3(0, 0, 0)),
                   low_left_corner(Vector3(0, 0, 0)),
                   horizontal_vec(Vector3(0, 0, 0)),
                   vertical_vec(Vector3(0, 0, 0)) {
}



Ray Camera::gen_ray(double _u, double _v) {
    Ray tmp_r(origin, low_left_corner + _u * horizontal_vec + _v * vertical_vec - origin);
    return {tmp_r};
}

Camera::Camera(const Vector3 &_look_from, const Vector3 &_look_at, const Vector3 &_view_up, const double &_vfov,
               const double &_aspect) {
    //vfov top to bottom in degrees & aspect = width / height
    double theta = _vfov * M_PI / 180;
    double half_height = tan(theta / 2);
    double half_width = _aspect * half_height;
    this->origin = _look_from;
    this->w = (_look_from-_look_at).normalize();
    this->u = cross(_view_up,this->w).normalize();
    this->v = cross(w,u).normalize();
    this->low_left_corner = origin - u * half_width - v * half_height - w;
    this->horizontal_vec = 2 * half_width * u;
    this->vertical_vec =  2 * half_height * v;


}



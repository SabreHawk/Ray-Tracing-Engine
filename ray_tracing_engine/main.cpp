#include <iostream>

#include "src/PNGMaster.h"
#include "src/Vector3.h"
#include "src/Ray.h"
#include "src/Scene.h"
#include "src/Sphere.h"
void render();
double hitSphere(const Vector3&,double,const Ray&);


int main() {
    std::cout << "Hello, World!" << std::endl;
    Scene tmp_scene;
    Sphere tmp_sphere(Vector3(0,0,0),3);
    cout << &tmp_sphere << endl;
    tmp_scene.addObject(&tmp_sphere);
    std::cout << "End" << endl;
    return 0;
}

void render() {
    unsigned int height = 256;
    unsigned int width = 256;
    Vector3 origin_vec(128,128, 10);
    Vector3 vertical_vec(0, 1, 0);
    Vector3 horizontal_vec(1, 0, 0);
    PNGMaster tmp_pic(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            Vector3 tmp_pos = j * Vector3(1,0,0) + i * Vector3(0,1,0);
            Ray tmp_r(origin_vec, (tmp_pos-origin_vec).normalize());
            double t = hitSphere(Vector3(128,128,-100),80,tmp_r);
            Vector3 tmp_color =  Vector3(128,64,32);
            if (t > 0.0) {
                tmp_color = (Vector3(tmp_r.targetPos(t)-Vector3(128,128,-100))+Vector3(64,64,64)).normalize() * 256;
            }
            tmp_pic.setPixel(i, j, int(tmp_color.r()), int(tmp_color.g()),
                             int(tmp_color.b()));
        }

    }
    tmp_pic.genPNG("test1.png");

}

double hitSphere(const Vector3 &_center, double _r, const Ray &_ray) {
    Vector3 oc = _ray.origin() - _center;
    double a = dot(_ray.direction(), _ray.direction());
    double b = 2.0 * dot(oc, _ray.direction());
    double c = dot(oc, oc) - _r * _r;
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }

}
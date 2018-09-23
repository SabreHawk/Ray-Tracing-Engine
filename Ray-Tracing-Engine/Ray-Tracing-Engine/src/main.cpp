#include <iostream>

#include "PNGMaster.h"
#include "Vector3.h"
#include "Ray.h"
#include "Scene.h"
#include "Sphere.h"
void render();
double hitSphere(const Vector3&,double,const Ray&);


int main() {

    std::cout << "Hello, World!" << std::endl;
	Vector3 tmp_v(1, 1, 1);
	cout << tmp_v.normalize() << endl;
    render();
    std::cout << "End" << endl;
    return 0;
}

void render() {
    unsigned int height = 100;
    unsigned int width = 200;
    Vector3 lower_left_corner(-2.0,-1.0,-1.0);
    Vector3 vertical_vec(0.0, 2.0, 0.0);
    Vector3 horizontal_vec(4.0, 0, 0);
    Vector3 origin_vec(0.0,0.0,0.0);
    Scene tmp_scene;
    tmp_scene.addObject(new Sphere(Vector3(0,0,-1),0.5));
    tmp_scene.addObject(new Sphere(Vector3(0,-100.5,-1),100));
    PNGMaster tmp_pic(height, width);
    for (int i = height-1; i >=0 ; --i) {
        for (int j = 0; j < (int)width; ++j) {
            double u = float(i)/float(width);
            double v = float(j)/float(height);
            Ray * tmp_ray = new Ray(origin_vec,lower_left_corner+u*horizontal_vec+v*vertical_vec);
//            tmp_pic.setPixel(i, j, int(tmp_color.r()), int(tmp_color.g()),
//                             int(tmp_color.b()));
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
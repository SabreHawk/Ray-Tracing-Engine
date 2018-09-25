#include <iostream>

#include "PNGMaster.h"
#include "Vector3.h"
#include "Ray.h"
#include "Scene.h"
#include "Sphere.h"
#include "Camera.h"

void render();

Vector3 color(const Ray &, const Scene &);

int main() {

    std::cout << "Hello, World!" << std::endl;
    render();
    std::cout << "End" << endl;
    return 0;
}

void render() {
    unsigned int height = 1080;
    unsigned int width = 1920;
    int ray_num = 100;
    Vector3 lower_left_corner(-2.0, -1.0, -1.0);
    Vector3 vertical_vec(0.0, 2.0, 0.0);
    Vector3 horizontal_vec(4.0, 0, 0);
    Vector3 origin_vec(0.0, 0.0, 0.0);
    Scene tmp_scene;
    tmp_scene.addObject(new Sphere(Vector3(0, 0, -1), 0.5));
    tmp_scene.addObject(new Sphere(Vector3(0, -100.5, -1), 100));
    PNGMaster tmp_pic(height, width);
    Camera tmp_camera(origin_vec, lower_left_corner, horizontal_vec, vertical_vec);
    for (int i = 0; i < height; ++i) {
        cout << (float) i / height << endl;
        for (int j = 0; j < (int) width; ++j) {
            Vector3 total_color(0, 0, 0);
            for (int k = 0; k < ray_num; ++k) {
                double u = float(j + drand48()) / float(width);
                double v = float(i + drand48()) / float(height);
                Ray tmp_ray = tmp_camera.gen_ray(u, v);
                Vector3 tmp_color = 255.99 * color(tmp_ray, tmp_scene);
                total_color += tmp_color;
            }
            total_color /= ray_num;
            tmp_pic.setPixel(j, i, int(total_color.r()), int(total_color.g()),
                             int(total_color.b()));
        }

    }

    tmp_pic.genPNG("test.png");
}

Vector3 color(const Ray &_r, const Scene &_s) {
    hitInfo tmp_info;
    if (_s.hit(_r, 0.0, MAXFLOAT, tmp_info)) {
        return 0.5 * Vector3(tmp_info.normal.x() + 1, tmp_info.normal.y() + 1, tmp_info.normal.z() + 1);
    } else {
        Vector3 unit_vec = _r.direction();
        double t = 0.5 * (unit_vec.y() + 1.0);
        return (1.0 - t) * Vector3(1, 1, 1) + t * Vector3(0.5, 0.7, 1.0);
    }
}

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <functional>
#include <thread>
#include <pthread.h>
#include "PNGMaster.h"
#include "Vector3.h"
#include "Ray.h"
#include "Scene.h"
#include "Sphere.h"
#include "Camera.h"

void render0();

void render1();

void color(const Ray &, const Scene &, Vector3 &);

void cal_color(const int &, const int &, Vector3 &);

Vector3 random_in_unit_sphere();

unsigned int height = 108;
unsigned int width = 192;
int ray_num = 1000;
Vector3 lower_left_corner(-2.0, -1.0, -1.0);
Vector3 vertical_vec(0.0, 2.0, 0.0);
Vector3 horizontal_vec(4.0, 0, 0);
Vector3 origin_vec(0.0, 0.0, 0.0);
Scene tmp_scene;
Camera tmp_camera(origin_vec, lower_left_corner, horizontal_vec, vertical_vec);
PNGMaster tmp_pic(height, width);
int main() {

    std::cout << "Rendering" << std::endl;
    render1();
    return 0;
}

void render0() {

    tmp_scene.addObject(new Sphere(Vector3(0, 0, -1), 0.5));
    tmp_scene.addObject(new Sphere(Vector3(0, -100.5, -1), 100));
    PNGMaster tmp_pic(height, width);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < height; ++i) {
        if (i % 2 == 0) {
            cout << flush << '\r';
            printf("%.2lf%%", i * 100.0 / height);
        }
        for (int j = 0; j < (int) width; ++j) {
            Vector3 total_color(0, 0, 0);
            for (int k = 0; k < ray_num; ++k) {
                double u = float(j + drand48()) / float(width);
                double v = float(i + drand48()) / float(height);
                Ray tmp_ray = tmp_camera.gen_ray(u, v);
                Vector3 tmp_color;
                std::thread tmp_t(color, tmp_ray, tmp_scene, std::ref(tmp_color));
                tmp_t.detach();
//                color(tmp_ray,tmp_scene,tmp_color);
                tmp_color *= 255.99;
                total_color += tmp_color;
            }
            total_color /= ray_num;
            tmp_pic.setPixel(j, i, int(total_color.r()), int(total_color.g()),
                             int(total_color.b()));
        }

    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << flush << '\r' << "Render Completed - Time Consumed :"
              << (double) duration.count() / std::chrono::microseconds::period::den << 's' << endl;
    std::cout << "Generating Picture ...";
    tmp_pic.genPNG("test.png");
    std::cout << flush << '\r' << "Picture Generated Successfully" << endl;
}

void render1() {

    tmp_scene.addObject(new Sphere(Vector3(0, 0, -1), 0.5));
    tmp_scene.addObject(new Sphere(Vector3(0, -100.5, -1), 100));

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < height; ++i) {
        if (i % 2 == 0) {
            cout << flush << '\r';
            printf("%.2lf%%", i * 100.0 / height);
        }
        for (int j = 0; j < (int) width; ++j) {
            Vector3 total_color(0, 0, 0);
//            std::thread tmp_t(cal_color,i,j,std::ref(total_color));
//            tmp_t.detach();
            cal_color(i,j,total_color);

        }

    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << flush << '\r' << "Render Completed - Time Consumed :"
              << (double) duration.count() / std::chrono::microseconds::period::den << 's' << endl;
    std::cout << "Generating Picture ...";
    tmp_pic.genPNG("test.png");
    std::cout << flush << '\r' << "Picture Generated Successfully" << endl;
}

void cal_color(const int &_i, const int &_j, Vector3 &_c) {
    for (int k = 0; k < ray_num; ++k) {
        double u = float(_j + drand48()) / float(width);
        double v = float(_i + drand48()) / float(height);
        Ray tmp_ray = tmp_camera.gen_ray(u, v);
        Vector3 tmp_color;
//        std::thread tmp_t(color, tmp_ray, tmp_scene, std::ref(tmp_color));
//        tmp_t.detach();
        color(tmp_ray,tmp_scene,tmp_color);
        _c += tmp_color;
    }
    _c /= ray_num;
    _c  = Vector3(sqrt(_c.r()),sqrt(_c.g()),sqrt(_c.b()));
    _c *= 255.99;
    tmp_pic.setPixel(_j, _i, int(_c.r()), int(_c.g()),
                     int(_c.b()));
}

void color(const Ray &_r, const Scene &_s, Vector3 &_c) {
    hitInfo tmp_info;
    if (_s.hit(_r, 0.001, MAXFLOAT, tmp_info)) {
        Vector3 next_dir = tmp_info.pos + tmp_info.normal + random_in_unit_sphere() - tmp_info.pos;
        color(Ray(tmp_info.pos, next_dir), _s, std::ref(_c));
        _c *= 0.5;
    } else {
        Vector3 unit_vec = _r.direction();
        double t = 0.5 * (unit_vec.y() + 1.0);
        _c = (1.0 - t) * Vector3(1, 1, 1) + t * Vector3(0.2, 0.5, 1.0);
    }
}

Vector3 random_in_unit_sphere() {
    Vector3 p;
    do {
        p = 2.0 * Vector3(drand48(), drand48(), drand48()) - Vector3(1, 1, 1);
    } while (p.length() * p.length() >= 1.0);
    return p;
}

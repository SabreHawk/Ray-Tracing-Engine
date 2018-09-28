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
#include "Material.h"
#include "HitInfo.h"
#include "Lambertian.h"
#include "Metal.h"
#include "Dielectric.h"

void render0();

void render1();

Vector3 color(const Ray &, const Scene &, int);

void cal_color(const int &, const int &);


unsigned int height = 108;
unsigned int width = 192;
int ray_num = 100  ;
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

void render1() {

    tmp_scene.addObject(new Sphere(Vector3(0, 0, -1), 0.5, new Lambertian(Vector3(0.8, 0.3, 0.3))));
    tmp_scene.addObject(new Sphere(Vector3(0, -100.5, -1), 100, new Lambertian(Vector3(0.8, 0.8, 0.0))));
    tmp_scene.addObject(new Sphere(Vector3(1, 0, -1), 0.5, new Metal(Vector3(0.8, 0.6, 0.2),0)));
    tmp_scene.addObject(new Sphere(Vector3(-1, 0, -1), 0.5, new Dielectric(1.5)));
    tmp_scene.addObject(new Sphere(Vector3(-1, 0, -1), -0.45, new Dielectric(1.5)));

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
            cal_color(i, j);

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

void cal_color(const int &_i, const int &_j) {
    Vector3 tmp_color;
    for (int k = 0; k < ray_num; ++k) {
        double u = float(_j + drand48()) / float(width);
        double v = float(_i + drand48()) / float(height);
        Ray tmp_ray = tmp_camera.gen_ray(u, v);

//        std::thread tmp_t(color, tmp_ray, tmp_scene, std::ref(tmp_color));
//        tmp_t.detach();

        tmp_color += color(tmp_ray, tmp_scene, 0);;
    }
    tmp_color /= ray_num;
    tmp_color = Vector3(sqrt(tmp_color.r()), sqrt(tmp_color.g()), sqrt(tmp_color.b()));
    tmp_color *= 255.99;
    tmp_pic.setPixel(_j, _i, int(tmp_color.r()), int(tmp_color.g()),
                     int(tmp_color.b()));
}

Vector3 color(const Ray &_r, const Scene &_s, int _d) {
    HitInfo tmp_info;
    if (_s.hit(_r, 0.001, MAXFLOAT, tmp_info)) {//if hit object
        Ray scatter_ray;
        Vector3 attenuation_vec;
        if (_d < 50 && tmp_info.material_ptr->scatter(_r, tmp_info, attenuation_vec, scatter_ray)) {
            return attenuation_vec * color(scatter_ray, _s, _d + 1);
        } else {
            return Vector3(0, 0, 0);
        }
    } else {
        Vector3 unit_vec = _r.direction();
        double t = 0.5 * (unit_vec.y() + 1.0);
        return (1.0 - t) * Vector3(1, 1, 1) + t * Vector3(0.5, 0.7, 1.0);
    }
}


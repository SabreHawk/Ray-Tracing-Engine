#include <iostream>
#include <unistd.h>
#include <chrono>
#include <functional>
#include <thread>
#include <pthread.h>
#include <iomanip>
#include <cfloat>
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
#include "limits.h"
#include <cfloat>
void render0();

void render1();

Vector3 color(const Ray &, const Scene &, int);

void cal_color(const int &, const int &);

void random_scene();

unsigned int height = 1080;
unsigned int width = 1920;
int ray_num = 1000;
Vector3 lower_left_corner(-2.0, -1.0, -1.0);
Vector3 vertical_vec(0.0, 2.0, 0.0);
Vector3 horizontal_vec(4.0, 0, 0);
Vector3 origin_vec(0.0, 0.0, 0.0);
Scene tmp_scene;
//Camera Scene 01
//Camera tmp_camera(origin_vec, lower_left_corner, horizontal_vec, vertical_vec);
//Camera Scene 02
//Camera tmp_camera(90, double(width)/height);
Vector3 look_from(13, 2, 3);
Vector3 look_at(0, 0, 0);
float focus_dis = (look_from - look_at).length();
Camera tmp_camera(look_from, look_at, Vector3(0, 1, 0), 20, float(width) / float(height), 0.1,
                  10);
PNGMaster tmp_pic(height, width);

int main() {

    std::cout << "Rendering" << std::endl;
    render1();
    return 0;
}

void random_scene() {
    tmp_scene.addObject(new Sphere(Vector3(0, -1000, 0), 1000, new Lambertian(Vector3(0.5, 0.5, 0.5))));
    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            double material_probability = drand48();
            Vector3 tmp_center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
            if ((tmp_center - Vector3(4, 0.2, 0)).length() > 0.9) {
                if (material_probability < 0.8) {
                    tmp_scene.addObject(new Sphere(tmp_center, 0.2, new Lambertian(
                            Vector3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48()))));
                } else if (material_probability < 0.95) {
                    tmp_scene.addObject(new Sphere(tmp_center, 0.2, new Metal(
                            Vector3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()), 0.5 * (1 + drand48())),
                            0.5 * drand48())));
                } else {
                    tmp_scene.addObject(new Sphere(tmp_center, 0.2, new Dielectric(1.5)));
                }
            }
        }
    }
    tmp_scene.addObject(new Sphere(Vector3(0, 1, 0), 1.0, new Dielectric(1.5)));
    tmp_scene.addObject(new Sphere(Vector3(-4, 1, 0), 1.0, new Lambertian(Vector3(0.4, 0.2, 0.1))));
    tmp_scene.addObject(new Sphere(Vector3(4, 1, 0), 1.0, new Metal(Vector3(0.7, 0.6, 0.5), 0.0)));
}

void render1() {
//    Scene 01
//    tmp_scene.addObject(new Sphere(Vector3(0, 0, -1), 0.5, new Lambertian(Vector3(0.8, 0.3, 0.3))));
//    tmp_scene.addObject(new Sphere(Vector3(0, -100.5, -1), 100, new Lambertian(Vector3(0.8, 0.8, 0.0))));
//    tmp_scene.addObject(new Sphere(Vector3(1, 0, -1), 0.5, new Metal(Vector3(0.8, 0.6, 0.2), 0)));
//    tmp_scene.addObject(new Sphere(Vector3(-1, 0, -1), 0.5, new Dielectric(1.5)));
//    tmp_scene.addObject(new Sphere(Vector3(-1, 0, -1), -0.45, new Dielectric(1.5)));
    //Scene02 - Test Camera
//    double r = cos(M_PI/4);
//    tmp_scene.addObject(new Sphere(Vector3(-r,0,-1),r,new Lambertian(Vector3(0,0,1))));
//    tmp_scene.addObject(new Sphere(Vector3(r,0,-1),r,new Lambertian(Vector3(1,0,0))));
    random_scene();

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    for (int i = 0; i < height; ++i) {
        if (i % 2 == 0) {
            cout << flush << '\r';
            printf("%.2lf%%", i * 100.0 / height);
            end = std::chrono::system_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cout << "    "<<(double) duration.count() / std::chrono::microseconds::period::den /60<< "min";

        }
        for (int j = 0; j < (int) width; ++j) {
            Vector3 total_color(0, 0, 0);
//            std::thread tmp_t(cal_color,i,j,std::ref(total_color));
//            tmp_t.detach();
            cal_color(i, j);

        }

    }
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << flush << '\r' << "Render Completed - Time Consumed :"
              << (double) duration.count() / std::chrono::microseconds::period::den/60 << "min" << endl;
    std::cout << "Generating Picture ...";
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%F %T");
    std::string str = ss.str();
    char tmp_name[100];

    strcpy(tmp_name, ss.str().c_str());
    strcat(tmp_name, ".png");
    tmp_name[13] = '-';
    tmp_name[16] = '-';
    tmp_pic.genPNG(tmp_name);
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
    if (_s.hit(_r, 0.001, DBL_MAX, tmp_info)) {//if hit object
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


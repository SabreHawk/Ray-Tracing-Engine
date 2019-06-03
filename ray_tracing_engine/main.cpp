#include "Camera.h"
#include "Dielectric.h"
#include "Lambertian.h"
#include "Metal.h"
#include "PNGMaster.h"
#include "Ray.h"
#include "Scene.h"
#include "Sphere.h"
#include "Vector3.h"
#include <cfloat>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

void render0();

void render1();

Vector3 color(const Ray &, const Scene &, int);

void cal_color(const int &, const int &);

void random_scene();

void read_fluid(const char *_file_name);

unsigned int height = 270;
unsigned int width = 480;
int ray_num = 10;
Vector3 lower_left_corner(-2.0, -1.0, -1.0);
Vector3 vertical_vec(0.0, 2.0, 0.0);
Vector3 horizontal_vec(4.0, 0, 0);
Vector3 origin_vec(0.0, 0.0, 0.0);
Scene tmp_scene;
// Camera Scene 01
// Camera tmp_camera(origin_vec, lower_left_corner, horizontal_vec,
// vertical_vec); Camera Scene 02 Camera tmp_camera(90, double(width)/height);
Vector3 look_from(13, 2, 3);
Vector3 look_at(0, 0, 0);
float focus_dis = (look_from - look_at).length();
Camera tmp_camera(look_from, look_at, Vector3(0, 1, 0), 30,
                  float(width) / float(height), 0.1, focus_dis, 0, 0);
PNGMaster tmp_pic(height, width);
string fluid_dataset_name;
int p_num;
int f_num;
float f_interval;
float animation_duration;

int main() {

  std::cout << "Rendering : " << width << " - " << height << std::endl;
  render1();
  //   "/home/zhiquan/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/"
  //           "2.txt");
  return 0;
}

void random_scene() {
  tmp_scene.addObject(new Sphere(Vector3(0, -1000, 0), 1000,
                                 new Lambertian(Vector3(0.5, 0.5, 0.5))));
  for (int a = -11; a < 11; ++a) {
    for (int b = -11; b < 11; ++b) {
      double material_probability = drand48();
      Vector3 tmp_center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
      if ((tmp_center - Vector3(4, 0.2, 0)).length() > 0.9) {
        if (material_probability < 0.8) {
          Sphere *tmp_sphere =
              new Sphere(tmp_center, 0.2,
                         new Lambertian(Vector3(drand48() * drand48(),
                                                drand48() * drand48(),
                                                drand48() * drand48())));
          // tmp_sphere->add_node(tmp_center,0);
          // tmp_sphere->add_node(tmp_center+Vector3(0,0.5*drand48(),0),1.0);
          tmp_scene.addObject(tmp_sphere);
        } else if (material_probability < 0.95) {
          tmp_scene.addObject(new Sphere(
              tmp_center, 0.2,
              new Metal(Vector3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()),
                                0.5 * (1 + drand48())),
                        0.5 * drand48())));
        } else {
          tmp_scene.addObject(new Sphere(tmp_center, 0.2, new Dielectric(1.5)));
        }
      }
    }
  }
  tmp_scene.addObject(new Sphere(Vector3(0, 1, 0), 1.0, new Dielectric(1.5)));
  tmp_scene.addObject(new Sphere(Vector3(-4, 1, 0), 1.0,
                                 new Lambertian(Vector3(0.4, 0.2, 0.1))));
  tmp_scene.addObject(new Sphere(Vector3(4, 1, 0), 1.0,
                                 new Metal(Vector3(0.7, 0.6, 0.5), 0.0)));
}

void render1() {
  random_scene();
  // read_fluid("/home/zhiquan/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/2.txt");
  /*
    std::ifstream in_stream(_file_name);
    std::string tmp_line;
    if (in_stream) {
      // read file name
      getline(in_stream, tmp_line);
      std::stringstream s_stream(tmp_line);
      s_stream >> fluid_dataset_name;
      // std::cout << fluid_dataset_name << endl << endl;
      // read fluid parameters
      getline(in_stream, tmp_line);
      // std::cout << "tmp " << tmp_line << endl;
      s_stream.clear();
      s_stream.str(tmp_line);
      s_stream >> p_num;
      s_stream >> f_num;
      s_stream >> f_interval;
      s_stream >> animation_duration;
      // std::cout << p_num << endl;
      // read each frame info
      for (int frame_index = 0; frame_index < 265; ++frame_index) {
        auto start = std::chrono::system_clock::now();
        auto end = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        getline(in_stream, tmp_line);
        s_stream.clear();
        s_stream.str(tmp_line);
        int tmp_f_index;
        s_stream >> tmp_f_index;
        if (tmp_f_index <= 184) {
          cout << tmp_f_index + 1 << " skiped" << endl;
          for (int p_index = 0; p_index < p_num; ++p_index) {
            getline(in_stream, tmp_line);
          }
          continue;
        }

        for (int p_index = 0; p_index < p_num; ++p_index) {
          getline(in_stream, tmp_line);
          tmp_line = tmp_line.replace(tmp_line.find('('), 1, "");
          tmp_line = tmp_line.replace(tmp_line.find(')'), 1, "");
          tmp_line = tmp_line.replace(tmp_line.find(','), 1, " ");
          tmp_line = tmp_line.replace(tmp_line.find(','), 1, " ");
          // std::cout << tmp_line << endl;
          s_stream.clear();
          s_stream.str(tmp_line);
          Sphere tmp_s;
          int tmp_index;
          s_stream >> tmp_index;
          float tmp_vec[3];
          for (float &ii : tmp_vec) {
            s_stream >> ii;
          }
          // std::cout << Vector3(tmp_vec[0],tmp_vec[1],tmp_vec[2])<<endl;
          tmp_scene.addObject(new Sphere(
              Vector3(tmp_vec[0], tmp_vec[1], tmp_vec[2]), 10,
              new Metal(Vector3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()),
                                0.5 * (1 + drand48())),
                        0.5 * drand48())));
        }
  */
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  for (int i = 0; i < height; ++i) {
    if (i % 2 == 0) {
      cout << flush << '\r';
      printf("%.2lf%%", i * 100.0 / height);
      end = std::chrono::system_clock::now();
      duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      printf("\t%.2f min", (double)duration.count() /
                               std::chrono::microseconds::period::den / 60);
      printf("\tEstimated Time : %.2f min",
             (double)(height - i) / i * (double)duration.count() /
                 std::chrono::microseconds::period::den / 60);
    }
    for (int j = 0; j < (int)width; ++j) {
      Vector3 total_color(0, 0, 0);
      //            std::thread tmp_t(cal_color,i,j,std::ref(total_color));
      //            tmp_t.detach();
      cal_color(i, j);
    }
  }
  std::cout << "Generating Picture: " << width << "*" << height << endl;
  ;
  auto t =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::stringstream ss;
  ss << std::put_time(std::localtime(&t), "%F %T");
  std::string tmp_str = ss.str();
  //   ss.clear();
  //   ss << frame_index;
  char tmp_name[100];

  strcpy(tmp_name, tmp_str.c_str());
  strcat(tmp_name, "test.png");
  // tmp_name[13] = '-';
  // tmp_name[16] = '-';
  // Timer
  end = std::chrono::system_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << flush << '\r' << "Render Completed - Time Consumed :"
            << (double)duration.count() /
                   std::chrono::microseconds::period::den / 60
            << "min" << endl;
  tmp_pic.genPNG(tmp_name);
  //   tmp_scene.clear();
  //   std::cout << flush << '\r' << frame_index + 1 << "/" << f_num
  //             << " Picture Generated Successfully " << endl;
}

void read_fluid(const char *_file_name) {
  std::ifstream in_stream(_file_name);
  std::string tmp_line;
  if (in_stream) {
    // read file name
    getline(in_stream, tmp_line);
    std::stringstream s_stream(tmp_line);
    s_stream >> fluid_dataset_name;
    std::cout << fluid_dataset_name << endl << endl;
    // read fluid parameters
    getline(in_stream, tmp_line);
    std::cout << "tmp " << tmp_line << endl;
    s_stream.clear();
    s_stream.str(tmp_line);
    s_stream >> p_num;
    s_stream >> f_interval;
    s_stream >> animation_duration;
    std::cout << p_num << endl;
    // read each frame info
    for (int i = 0; i < 1; ++i) {
      getline(in_stream, tmp_line);
      s_stream.clear();
      s_stream.str(tmp_line);
      for (int j = 0; j < p_num; ++j) {
        getline(in_stream, tmp_line);
        tmp_line = tmp_line.replace(tmp_line.find('('), 1, "");
        tmp_line = tmp_line.replace(tmp_line.find(')'), 1, "");
        tmp_line = tmp_line.replace(tmp_line.find(','), 1, " ");
        tmp_line = tmp_line.replace(tmp_line.find(','), 1, " ");
        std::cout << tmp_line << endl;
        s_stream.clear();
        s_stream.str(tmp_line);
        Sphere tmp_s;
        int tmp_index;
        s_stream >> tmp_index;
        float tmp_vec[3];
        for (float &ii : tmp_vec) {
          s_stream >> ii;
        }
        std::cout << Vector3(tmp_vec[0], tmp_vec[1], tmp_vec[2]) << endl;
        tmp_scene.addObject(new Sphere(
            Vector3(tmp_vec[0], tmp_vec[1], tmp_vec[2]), 16,
            new Metal(Vector3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()),
                              0.5 * (1 + drand48())),
                      0.5 * drand48())));
      }
    }
  } else {
    std::cout << "ERROR : No File Named <" << _file_name << ">" << std::endl;
  }
}

void cal_color(const int &_i, const int &_j) {
  Vector3 tmp_color;
  for (int k = 0; k < ray_num; ++k) {
    double u = float(_j + drand48()) / float(width);
    double v = float(_i + drand48()) / float(height);
    Ray tmp_ray = tmp_camera.gen_ray(u, v);
    tmp_color += color(tmp_ray, tmp_scene, 0);
    ;
  }
  tmp_color /= ray_num;
  tmp_color = Vector3(sqrt(tmp_color.r()), sqrt(tmp_color.g()),
                      sqrt(tmp_color.b())); // gamma corrected with gamma 2
  tmp_color *= 255.99;
  tmp_pic.setPixel(_j, _i, int(tmp_color.r()), int(tmp_color.g()),
                   int(tmp_color.b()));
}

Vector3 color(const Ray &_r, const Scene &_s, int _d) {
  HitInfo tmp_info;
  if (_s.hit(_r, 0.001, DBL_MAX, tmp_info)) { // if hit object
    Ray scatter_ray;
    Vector3 attenuation_vec;
    if (_d < 20 && tmp_info.material_ptr->scatter(_r, tmp_info, attenuation_vec,
                                                  scatter_ray)) {
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

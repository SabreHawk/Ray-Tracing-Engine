#include "pngmaster.h"
#include "ray.cuh"
#include "scene.cuh"
#include "shpere.cuh"
#include "vector3.cuh"
#include <float.h>
#include <iostream>
#include <time.h>
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ vector3 color(const ray &_r, scene **_tmp_scene) {
  hitinfo tmp_info;
  if ((*_tmp_scene)->hit(_r, 0.0, FLT_MAX, tmp_info)) {
    return 0.5 * vector3(tmp_info.normal.x() + 1.0f, tmp_info.normal.y() + 1.0f,
                         tmp_info.normal.z() + 1.0f);
  }

  vector3 unit_direction = _r.direction().normalize();
  float t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * vector3(1, 1, 1) + t * vector3(0.0, 0.0, 0.0);
}

__global__ void render(vector3 *fb, int max_x, int max_y,
                       vector3 lower_left_corner, vector3 horizontal,
                       vector3 vertical, vector3 origin, scene **tmp_scene) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray tmp_r(origin, lower_left_corner + u * horizontal + v * vertical);
  fb[pixel_index] = color(tmp_r, tmp_scene);
}

__global__ void init_scene(object **objs, scene **tmp_scene) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(objs) = new sphere(vector3(0.0f, 0.0f, -1.0f), 0.5f);
    *(objs + 1) = new sphere(vector3(0.0f, -100.5f, -1.0f), 0.5f);
    *tmp_scene = new scene(objs, 2);
  }
}
int main() {
  int nx = 1200;
  int ny = 600;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vector3);

  // allocate FB
  vector3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // init scene
  const int obj_nums = 2;
  scene **d_tmp_scene;
  checkCudaErrors(cudaMalloc((void **)&d_tmp_scene, sizeof(scene *)));
  object **d_objs;
  checkCudaErrors(cudaMalloc((void **)&d_objs, sizeof(object *) * obj_nums));
  init_scene<<<1, 1>>>(d_objs, d_tmp_scene);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny, vector3(-2.0f, -1.0f, -1.0f),
                              vector3(4.0f, 0.0f, 0.0f), vector3(0.0, 2.0, 0.0),
                              vector3(0.0f, 0.0f, 0.0f), d_tmp_scene);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  float timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  pngmaster myImage(ny, nx);
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      vector3 tmp_vec = fb[pixel_index] * 255.99f;
      myImage.set_pixel(i, j, tmp_vec.r(), tmp_vec.g(), tmp_vec.b());
    }
  }

  string file_name = "test" + std::to_string(timer_seconds) + ".png";
  myImage.output(file_name.c_str());
  std::cerr << "render finished" << std::endl;

  checkCudaErrors(cudaFree(fb));
}
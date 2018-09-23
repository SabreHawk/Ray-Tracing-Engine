cmake_minimum_required(VERSION 3.10)
project(ray_tracing_engine)

set(CMAKE_CXX_STANDARD 14)
add_executable(ray_tracing_engine  main.cpp Object.h Scene.h Scene.cpp Ray.h Sphere.h Sphere.cpp Vector3.cpp Vector3.h)
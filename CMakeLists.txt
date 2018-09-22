cmake_minimum_required(VERSION 3.5)
project(Ray_Tracing_Engine)

set(CMAKE_CXX_STANDARD 11)

add_executable(Ray_Tracing_Engine src/main.cpp src/Vector3.h src/Ray.h src/Scene.h src/Scene.cpp src/Sphere.h src/Sphere.cpp)
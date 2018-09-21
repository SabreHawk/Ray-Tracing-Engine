//
// Created by mySab on 2018/9/21.
//

#ifndef RAY_TRACING_ENGINE_SCENE_H
#define RAY_TRACING_ENGINE_SCENE_H


#include "Object.h"

class Scene {
private:
    Object **object_list;
    int object_num;
public:
    Scene();

};


#endif //RAY_TRACING_ENGINE_SCENE_H

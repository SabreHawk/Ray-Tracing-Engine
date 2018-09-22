//
// Created by mySab on 2018/9/21.
//

#ifndef RAY_TRACING_ENGINE_SCENE_H
#define RAY_TRACING_ENGINE_SCENE_H


#include "Object.h"

class Scene {
private:
    Object **object_list;
    int object_num = 0;
public:
    Scene() = default;
    bool addObject(Object *);

};

bool Scene::addObject(Object * _obj) {
    std::cout << object_num << std::endl;
    this->object_list[object_num++] = _obj;
    std::cout << 2 << std::endl;
    std::cout << _obj;
    std::cout << this->object_list[object_num-1];
}


#endif //RAY_TRACING_ENGINE_SCENE_H

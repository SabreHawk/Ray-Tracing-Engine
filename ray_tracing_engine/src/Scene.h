//
// Created by mySab on 2018/9/21.
//

#ifndef RAY_TRACING_ENGINE_SCENE_H
#define RAY_TRACING_ENGINE_SCENE_H


#include "Object.h"
#include <vector>

class Scene {
private:
    std::vector<Object *> object_list;
    int object_num = 0;
public:
    Scene() = default;
    bool addObject(Object *);

};

bool Scene::addObject(Object * _obj) {
    std::cout << object_num << std::endl;
    this->object_list.push_back(_obj);
    ++ object_num;
    std::cout << object_num << std::endl;
    _obj->dispInfo();
    this->object_list[object_num-1]->dispInfo();
    return true;
}


#endif //RAY_TRACING_ENGINE_SCENE_H

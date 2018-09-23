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
    int object_num;
public:
    Scene();
    bool addObject(Object *);

};



#endif //RAY_TRACING_ENGINE_SCENE_H

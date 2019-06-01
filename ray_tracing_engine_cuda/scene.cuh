//
// Created by mySab on 2019/06/01.
//

#ifndef RAY_TRACING_ENGINE_SCENE_CUDA_H
#define RAY_TRACING_ENGINE_SCENE_CUDA_H

#include "hitinfo.h"
#include "object.h"
#include <vector>

class Scene : public Object {
private:
  std::vector<object *> object_list;
  int object_num;

public:
  Scene();

  bool addObject(Object *);

  bool hit(const Ray &, double, double, HitInfo &) const override;

  void dispInfo() override;

  void clear();
};

#endif // RAY_TRACING_ENGINE_SCENE_H

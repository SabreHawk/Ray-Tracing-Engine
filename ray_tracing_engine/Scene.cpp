//
// Created by mySab on 2018/9/21.
//

#include "Scene.h"


Scene::Scene() :object_num(0) {

}

bool Scene::addObject(Object * _obj) {
	std::cout << object_num << std::endl;
	this->object_list.push_back(_obj);
	++object_num;
	std::cout << object_num << std::endl;
	_obj->dispInfo();
	this->object_list[object_num - 1]->dispInfo();
	return true;
}



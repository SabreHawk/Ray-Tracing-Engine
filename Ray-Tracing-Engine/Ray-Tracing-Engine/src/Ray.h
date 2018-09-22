//
// Created by ZhiquanWang on 2018/7/12.
//

#ifndef RAY_TRACING_ENGINE_RAY_H
#define RAY_TRACING_ENGINE_RAY_H
#include "Vector3.h"

class Ray {
private:
    Vector3 pos;
    Vector3 dir;
public:
    Ray() = default;

    Ray(double _p0,double _p1,double _p2,double _d0,double _d1,double _d2){
        pos.setVector(_p0,_p1,_p2);
        dir.setVector(_d0,_d1,_d2);
        this->dir = this->dir.normalize();
    }
    Ray(const Vector3 & _p,const Vector3 & _d){
        pos = _p;
        dir = _d;
    }

    Vector3 origin()  const{
        return pos;
    }

    Vector3 direction() const{
        return dir;
    }

    Vector3 targetPos(double _t) const{
        return pos + _t * dir;
    }
};


#endif //RAY_TRACING_ENGINE_RAY_H

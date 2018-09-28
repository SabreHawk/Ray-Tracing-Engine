//
// Created by mySab on 2018/9/27.
//

#include "Dielectric.h"
#include "HitInfo.h"

Dielectric::Dielectric(double _ref_index) :refractive_index(_ref_index){

}

bool Dielectric::scatter(const Ray &_in_ray, const HitInfo &_hit_info, Vector3 &_attenuation,
                         Ray &_scattered_ray) const {
    Vector3 reflected_dir= reflect(_in_ray.direction(),_hit_info.normal);
    _attenuation = Vector3(1.0,1.0,1.0);
    Vector3 outward_normal = -1*_hit_info.normal;
    double tmp_eta = this->refractive_index;
    double cosine_in =  tmp_eta * dot(_in_ray.direction(),_hit_info.normal);
//    std::cout << _in_ray.direction().length() << std::endl;
    if (dot(_in_ray.direction(),_hit_info.normal) <= 0){
        outward_normal = _hit_info.normal;
        tmp_eta = 1.0/tmp_eta;
        cosine_in = -1*dot(_in_ray.direction(),_hit_info.normal);
    }

//    std::cout << _hit_info.normal.length() << std::endl;
    Vector3 refracted_dir ;
    double reflected_prob = 0;
    if (refract(_in_ray.direction(),outward_normal,tmp_eta,refracted_dir) ){
        reflected_prob = reflection_coefficient(cosine_in,this->refractive_index);
    }else{
        reflected_prob = 1.0;
    }
    //std::cout << reflected_prob << std::endl;
    if(drand48() < reflected_prob){
        _scattered_ray = Ray(_hit_info.pos,reflected_dir);
    }else{
        _scattered_ray = Ray(_hit_info.pos,refracted_dir);
    }
    return true;
}

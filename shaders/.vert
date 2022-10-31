#version 450

layout(binding=0)uniform UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 proj;
}ubo;

layout(location=0)in vec3 pos;
layout(location=1)in vec3 col1;
layout(location=2)in vec3 col2;
layout(location=3)in vec3 col3;
layout(location=4)in vec3 col4;
layout(location=5)in vec3 color;

layout(location=0)out vec3 fragColor;

void main(){
    mat4 instance=mat4(vec4(col1,0.),vec4(col2,0.),vec4(col3,0.),vec4(col4,1.));
    gl_Position=ubo.proj*ubo.view*ubo.model*instance*vec4(pos,1.);
    fragColor=color;
}
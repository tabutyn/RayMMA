// SPDX-License-Identifier: MIT
//
// Minimal CUDA-result viewer. Rendering stays in the benchmark; this program
// only presents its three PPM outputs and has no ray-tracing SDK dependency.
#include <GLFW/glfw3.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct Image {
    int width=0,height=0;
    std::vector<unsigned char> pixels;
    GLuint texture=0;
};

static bool loadPpm(const char* path,Image* image) {
    std::ifstream input(path,std::ios::binary);
    std::string magic;
    int maximum=0;
    input>>magic>>image->width>>image->height>>maximum;
    input.get();
    if(!input||magic!="P6"||maximum!=255)return false;
    image->pixels.resize(size_t(image->width)*image->height*3);
    input.read(
        reinterpret_cast<char*>(image->pixels.data()),
        std::streamsize(image->pixels.size()));
    return bool(input);
}

static void upload(Image* image) {
    glGenTextures(1,&image->texture);
    glBindTexture(GL_TEXTURE_2D,image->texture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D,0,GL_RGB,image->width,image->height,0,
        GL_RGB,GL_UNSIGNED_BYTE,image->pixels.data());
}

static void drawImage(GLuint texture,float left,float right) {
    glBindTexture(GL_TEXTURE_2D,texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0,1);glVertex2f(left,-1);
    glTexCoord2f(1,1);glVertex2f(right,-1);
    glTexCoord2f(1,0);glVertex2f(right,1);
    glTexCoord2f(0,0);glVertex2f(left,1);
    glEnd();
}

int main(int argc,char** argv) {
    if(argc!=4)
        return std::fprintf(
            stderr,
            "usage: %s CUDA32.ppm CUDA_PACKET16.ppm WMMA_VALIDATED.ppm\n",
            argv[0]),2;
    Image images[3];
    for(int i=0;i<3;i++)if(!loadPpm(argv[i+1],&images[i]))
        return std::fprintf(stderr,"Cannot read P6 image: %s\n",argv[i+1]),1;
    if(!glfwInit())return std::fprintf(stderr,"GLFW initialization failed\n"),1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
    GLFWwindow* window=glfwCreateWindow(
        images[0].width*6,images[0].height*2,
        "RayMMA 2x: CUDA32 | CUDA-packet16 | WMMA-validated",nullptr,nullptr);
    if(!window) {
        glfwTerminate();
        return std::fprintf(stderr,"Cannot create OpenGL window\n"),1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    for(Image& image:images)upload(&image);
    glEnable(GL_TEXTURE_2D);
    glClearColor(.035f,.035f,.045f,1);
    while(!glfwWindowShouldClose(window)) {
        if(glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS)
            glfwSetWindowShouldClose(window,GLFW_TRUE);
        int width,height;
        glfwGetFramebufferSize(window,&width,&height);
        glViewport(0,0,width,height);
        glClear(GL_COLOR_BUFFER_BIT);
        drawImage(images[0].texture,-1.f,-1.f/3.f);
        drawImage(images[1].texture,-1.f/3.f,1.f/3.f);
        drawImage(images[2].texture,1.f/3.f,1.f);
        glDisable(GL_TEXTURE_2D);
        glColor3f(.3f,.8f,1.f);
        glBegin(GL_LINES);
        glVertex2f(-1.f/3.f,-1);glVertex2f(-1.f/3.f,1);
        glVertex2f(1.f/3.f,-1);glVertex2f(1.f/3.f,1);
        glEnd();
        glColor3f(1,1,1);
        glEnable(GL_TEXTURE_2D);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    for(Image& image:images)glDeleteTextures(1,&image.texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// SPDX-License-Identifier: MIT
// Live Coastal Cliff High renderer for CUDA32, packet16, and validated WMMA.
#include <GLFW/glfw3.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <jpeglib.h>
#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "production_bvh.h"

using namespace nvcuda;

constexpr int RENDER_W=1920;
constexpr int RENDER_H=1080;
constexpr int PACKET_RAYS=16;
constexpr int CUDA_THREADS=128;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__, \
                 cudaGetErrorString(e));std::exit(1); } } while(0)

struct Ray { Vec3 o,d; };
struct Hit { float t,u,v;int triangle; };
struct LocalFrame { Vec3 center;float scale; };
struct Camera { Vec3 origin,forward,right,up; };
struct TexCoords { float au,av,cu,cv,bu,bv; };

__host__ __device__ Vec3 add(Vec3 a,Vec3 b) {
    return {a.x+b.x,a.y+b.y,a.z+b.z};
}
__host__ __device__ Vec3 sub(Vec3 a,Vec3 b) {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
__host__ __device__ Vec3 mul(Vec3 a,float s) {
    return {a.x*s,a.y*s,a.z*s};
}
__host__ __device__ float dot(Vec3 a,Vec3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
__host__ __device__ Vec3 cross(Vec3 a,Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
__host__ __device__ Vec3 unit(Vec3 a) {
    float length2=dot(a,a);
    return length2>0?mul(a,rsqrtf(length2)):Vec3{0,0,1};
}

static Camera orbitCamera(float angle,float height) {
    Vec3 origin{2.2f*sinf(angle),height,2.2f*cosf(angle)};
    Vec3 target{0,0,0};
    Vec3 forward=unit(sub(target,origin));
    Vec3 right=unit(cross(forward,{0,1,0}));
    return {origin,forward,right,unit(cross(right,forward))};
}

__device__ Ray cameraRay(int pixel,int width,int height,Camera camera) {
    int x=pixel%width,y=pixel/width;
    float px=(2.f*(x+.5f)/width-1.f)*(float(width)/height)*.72f;
    float py=(1.f-2.f*(y+.5f)/height)*.72f;
    return {camera.origin,unit(add(
        camera.forward,add(mul(camera.right,px),mul(camera.up,py))))};
}

__device__ bool intersectAabb(
    Vec3 origin,Vec3 ray,Vec3 lo,Vec3 hi,float nearest) {
    float tMin=.001f,tMax=nearest;
    const float o[3]={origin.x,origin.y,origin.z};
    const float d[3]={ray.x,ray.y,ray.z};
    const float lower[3]={lo.x,lo.y,lo.z};
    const float upper[3]={hi.x,hi.y,hi.z};
    #pragma unroll
    for(int axis=0;axis<3;axis++) {
        if(fabsf(d[axis])<1e-12f) {
            if(o[axis]<lower[axis]||o[axis]>upper[axis])return false;
        } else {
            float a=(lower[axis]-o[axis])/d[axis];
            float b=(upper[axis]-o[axis])/d[axis];
            if(a>b){float q=a;a=b;b=q;}
            tMin=fmaxf(tMin,a);tMax=fminf(tMax,b);
            if(tMax<tMin)return false;
        }
    }
    return true;
}

__device__ bool intersectExact(
    Triangle q,Ray ray,float nearest,float* t,float* u,float* v) {
    Vec3 p=cross(ray.d,q.b),s=sub(ray.o,q.a);
    float determinant=dot(q.c,p);
    if(fabsf(determinant)<1e-7f)return false;
    float inverse=1.f/determinant;
    *u=dot(s,p)*inverse;
    Vec3 z=cross(s,q.c);
    *v=dot(ray.d,z)*inverse;
    *t=dot(q.b,z)*inverse;
    return *u>=0&&*v>=0&&*u+*v<=1&&*t>.001f&&*t<nearest;
}

__global__ __launch_bounds__(CUDA_THREADS,4) void traceCuda32(
    Hit* output,int width,int height,const Triangle* triangles,
    const WideNode* nodes,Camera camera) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    int rayCount=width*height;
    if(pixel>=rayCount)return;
    Ray ray=cameraRay(pixel,width,height,camera);
    Hit hit{1e30f,0,0,-1};
    int stack[32],stackTop=1;
    stack[0]=0;
    while(stackTop) {
        WideNode node=nodes[stack[--stackTop]];
        #pragma unroll
        for(int slot=0;slot<BVH_WIDTH;slot++) {
            if(slot>=node.childCount)break;
            if(!intersectAabb(
                   ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t))continue;
            if(node.child[slot]>=0) {
                if(stackTop<32)stack[stackTop++]=node.child[slot];
                continue;
            }
            int end=node.first[slot]+node.count[slot];
            for(int triangle=node.first[slot];triangle<end;triangle++) {
                float t,u,v;
                if(intersectExact(
                       triangles[triangle],ray,hit.t,&t,&u,&v))
                    hit={t,u,v,triangle};
            }
        }
    }
    output[pixel]=hit;
}

__global__ void traceCuda16(
    Hit* output,int width,int height,const Triangle* triangles,
    const WideNode* nodes,Camera camera) {
    int lane=threadIdx.x,pixel=blockIdx.x*PACKET_RAYS+lane;
    int rayCount=width*height;
    Ray ray{};
    if(lane<PACKET_RAYS&&pixel<rayCount)
        ray=cameraRay(pixel,width,height,camera);
    Hit hit{1e30f,0,0,-1};
    __shared__ int stack[64];
    __shared__ int stackTop;
    if(lane==0){stack[0]=0;stackTop=1;}
    __syncwarp();
    while(true) {
        int nodeIndex=-1;
        if(lane==0&&stackTop>0)nodeIndex=stack[--stackTop];
        nodeIndex=__shfl_sync(0xffffffffu,nodeIndex,0);
        if(nodeIndex<0)break;
        WideNode node=nodes[nodeIndex];
        for(int slot=0;slot<node.childCount;slot++) {
            bool rayHits=lane<PACKET_RAYS&&pixel<rayCount&&
                intersectAabb(
                    ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t);
            uint32_t mask=__ballot_sync(0xffffffffu,rayHits)&0xffffu;
            if(!mask)continue;
            if(node.child[slot]>=0) {
                if(lane==0&&stackTop<64)
                    stack[stackTop++]=node.child[slot];
                continue;
            }
            if(lane<PACKET_RAYS&&(mask&(1u<<lane))) {
                int end=node.first[slot]+node.count[slot];
                for(int triangle=node.first[slot];triangle<end;triangle++) {
                    float t,u,v;
                    if(intersectExact(
                           triangles[triangle],ray,hit.t,&t,&u,&v))
                        hit={t,u,v,triangle};
                }
            }
            __syncwarp();
        }
    }
    if(lane<PACKET_RAYS&&pixel<rayCount)output[pixel]=hit;
}

__device__ void tensorLeaf(
    Hit* hit,Ray ray,uint32_t rayMask,int lane,int firstTriangle,
    int triangleCount,const Triangle* triangles,const half* coefficients,
    const LocalFrame* frames,half* features,float* values) {
    int firstBatch=firstTriangle/4;
    int endBatch=(firstTriangle+triangleCount+3)/4;
    for(int batch=firstBatch;batch<endBatch;batch++) {
        LocalFrame frame=frames[batch];
        bool unsafe=!(frame.scale>0.f);
        if(lane<PACKET_RAYS) {
            Vec3 origin=unsafe?Vec3{0,0,0}:
                mul(sub(ray.o,frame.center),frame.scale);
            Vec3 moment=cross(origin,ray.d);
            float input[10]={
                1,origin.x,origin.y,origin.z,
                ray.d.x,ray.d.y,ray.d.z,moment.x,moment.y,moment.z};
            for(int k=0;k<10;k++)
                unsafe|=!isfinite(input[k])||fabsf(input[k])>65504.f;
            half* destination=features+lane*16;
            #pragma unroll
            for(int k=0;k<16;k++)
                destination[k]=__float2half(k<10&&!unsafe?input[k]:0.f);
        }
        __syncwarp();
        wmma::fragment<
            wmma::matrix_a,16,16,16,half,wmma::row_major> a;
        wmma::fragment<
            wmma::matrix_b,16,16,16,half,wmma::col_major> b;
        wmma::fragment<
            wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        wmma::load_matrix_sync(a,coefficients+size_t(batch)*256,16);
        wmma::load_matrix_sync(b,features,16);
        wmma::mma_sync(c,a,b,c);
        wmma::store_matrix_sync(values,c,16,wmma::mem_row_major);
        __syncwarp();
        if(lane<PACKET_RAYS&&(rayMask&(1u<<lane))) {
            #pragma unroll
            for(int local=0;local<4;local++) {
                int triangle=batch*4+local;
                if(triangle<firstTriangle||
                   triangle>=firstTriangle+triangleCount)continue;
                float nu=values[(local*4+1)*16+lane];
                float nv=values[(local*4+2)*16+lane];
                float de=values[(local*4+3)*16+lane];
                bool ambiguous=unsafe||!isfinite(nu)||!isfinite(nv)||
                               !isfinite(de)||fabsf(de)<1e-5f;
                float sign=de<0?-1.f:1.f;
                float d=sign*de,u=sign*nu,v=sign*nv;
                float tolerance=2.f*d;
                if(ambiguous||
                   (u>=-tolerance&&v>=-tolerance&&u+v<=d+tolerance)) {
                    float t,exactU,exactV;
                    if(intersectExact(
                           triangles[triangle],ray,hit->t,
                           &t,&exactU,&exactV))
                        *hit={t,exactU,exactV,triangle};
                }
            }
        }
        __syncwarp();
    }
}

__global__ void traceTensor16(
    Hit* output,int width,int height,const Triangle* triangles,
    const half* coefficients,const LocalFrame* frames,
    const WideNode* nodes,Camera camera) {
    int lane=threadIdx.x,pixel=blockIdx.x*PACKET_RAYS+lane;
    int rayCount=width*height;
    Ray ray{};
    if(lane<PACKET_RAYS&&pixel<rayCount)
        ray=cameraRay(pixel,width,height,camera);
    Hit hit{1e30f,0,0,-1};
    __shared__ __align__(32) half features[16*16];
    __shared__ __align__(32) float values[16*16];
    __shared__ int stack[64];
    __shared__ int stackTop;
    if(lane==0){stack[0]=0;stackTop=1;}
    __syncwarp();
    while(true) {
        int nodeIndex=-1;
        if(lane==0&&stackTop>0)nodeIndex=stack[--stackTop];
        nodeIndex=__shfl_sync(0xffffffffu,nodeIndex,0);
        if(nodeIndex<0)break;
        WideNode node=nodes[nodeIndex];
        for(int slot=0;slot<node.childCount;slot++) {
            bool rayHits=lane<PACKET_RAYS&&pixel<rayCount&&
                intersectAabb(
                    ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t);
            uint32_t mask=__ballot_sync(0xffffffffu,rayHits)&0xffffu;
            if(!mask)continue;
            if(node.child[slot]>=0) {
                if(lane==0&&stackTop<64)
                    stack[stackTop++]=node.child[slot];
            } else {
                tensorLeaf(
                    &hit,ray,mask,lane,node.first[slot],node.count[slot],
                    triangles,coefficients,frames,features,values);
            }
        }
    }
    if(lane<PACKET_RAYS&&pixel<rayCount)output[pixel]=hit;
}

__device__ int wrapCoordinate(int x,int size) {
    x%=size;
    return x<0?x+size:x;
}

__device__ unsigned char bilinearChannel(
    unsigned char a,unsigned char b,unsigned char c,unsigned char d,
    float fx,float fy) {
    float top=a+(b-a)*fx,bottom=c+(d-c)*fx;
    return static_cast<unsigned char>(top+(bottom-top)*fy+.5f);
}

__device__ uchar4 sampleTexture(
    const uchar4* texture,int width,int height,float u,float v) {
    float px=u*width-.5f,py=(1.f-v)*height-.5f;
    int x0=int(floorf(px)),y0=int(floorf(py));
    float fx=px-x0,fy=py-y0;
    int ax=wrapCoordinate(x0,width),bx=wrapCoordinate(x0+1,width);
    int ay=wrapCoordinate(y0,height),by=wrapCoordinate(y0+1,height);
    uchar4 aa=texture[ay*width+ax],ba=texture[ay*width+bx];
    uchar4 ab=texture[by*width+ax],bb=texture[by*width+bx];
    return make_uchar4(
        bilinearChannel(aa.x,ba.x,ab.x,bb.x,fx,fy),
        bilinearChannel(aa.y,ba.y,ab.y,bb.y,fx,fy),
        bilinearChannel(aa.z,ba.z,ab.z,bb.z,fx,fy),255);
}

__global__ void shade(
    uchar4* output,const Hit* hits,int width,int height,
    const TexCoords* texCoords,const uchar4* texture,
    int textureWidth,int textureHeight,Camera camera) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=width*height)return;
    Hit hit=hits[pixel];
    if(hit.triangle<0) {
        Ray ray=cameraRay(pixel,width,height,camera);
        float q=.5f+.5f*ray.d.y;
        output[pixel]=make_uchar4(18+25*q,28+38*q,45+70*q,255);
        return;
    }
    TexCoords uv=texCoords[hit.triangle];
    float u=uv.au+hit.u*(uv.cu-uv.au)+hit.v*(uv.bu-uv.au);
    float v=uv.av+hit.u*(uv.cv-uv.av)+hit.v*(uv.bv-uv.av);
    output[pixel]=sampleTexture(texture,textureWidth,textureHeight,u,v);
}

struct Bounds {
    Vec3 lo{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Vec3 hi{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()};
    void include(Vec3 p) {
        lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
        hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
    }
};

static void includeTriangle(Bounds* bounds,const Triangle& triangle) {
    bounds->include(triangle.a);
    bounds->include(add(triangle.a,triangle.c));
    bounds->include(add(triangle.a,triangle.b));
}

struct Scene {
    Triangle* triangles=nullptr;
    WideNode* nodes=nullptr;
    half* coefficients=nullptr;
    LocalFrame* frames=nullptr;
    TexCoords* texCoords=nullptr;
    uchar4* texture=nullptr;
    int trianglesCount=0,nodeCount=0,textureWidth=0,textureHeight=0;
};

static bool loadImage(
    const char* path,std::vector<uchar4>* pixels,int* width,int* height) {
    std::ifstream probe(path,std::ios::binary);
    unsigned char signature[2]{};
    probe.read(reinterpret_cast<char*>(signature),2);
    if(signature[0]==0xff&&signature[1]==0xd8) {
        FILE* file=std::fopen(path,"rb");
        if(!file)return false;
        jpeg_decompress_struct decoder{};
        jpeg_error_mgr errors{};
        decoder.err=jpeg_std_error(&errors);
        jpeg_create_decompress(&decoder);
        jpeg_stdio_src(&decoder,file);
        jpeg_read_header(&decoder,TRUE);
        decoder.out_color_space=JCS_RGB;
        jpeg_start_decompress(&decoder);
        *width=int(decoder.output_width);
        *height=int(decoder.output_height);
        pixels->resize(size_t(*width)*size_t(*height));
        std::vector<unsigned char> row(size_t(*width)*3);
        while(decoder.output_scanline<decoder.output_height) {
            JSAMPROW destination=row.data();
            jpeg_read_scanlines(&decoder,&destination,1);
            int y=int(decoder.output_scanline)-1;
            for(int x=0;x<*width;x++)
                (*pixels)[size_t(y)*size_t(*width)+x]=make_uchar4(
                    row[size_t(x)*3],row[size_t(x)*3+1],
                    row[size_t(x)*3+2],255);
        }
        jpeg_finish_decompress(&decoder);
        jpeg_destroy_decompress(&decoder);
        std::fclose(file);
        return true;
    }
    png_image image{};
    image.version=PNG_IMAGE_VERSION;
    if(!png_image_begin_read_from_file(&image,path))return false;
    image.format=PNG_FORMAT_RGBA;
    pixels->resize(PNG_IMAGE_SIZE(image)/4);
    bool ok=png_image_finish_read(
        &image,nullptr,pixels->data(),0,nullptr)!=0;
    *width=int(image.width);
    *height=int(image.height);
    png_image_free(&image);
    return ok;
}

static Scene loadScene(
    const char* geometryPath,const char* texturePath,int leafTriangles) {
    auto begin=std::chrono::steady_clock::now();
    std::ifstream input(geometryPath,std::ios::binary);
    char magic[8]{};
    uint32_t count=0;
    input.read(magic,8);
    input.read(reinterpret_cast<char*>(&count),4);
    if(!input||std::memcmp(magic,"BRTRI003",8)||!count) {
        std::fprintf(stderr,"Invalid Coastal Cliff High cache: %s\n",geometryPath);
        std::exit(1);
    }
    std::vector<Triangle> triangles(count);
    std::vector<TexCoords> sourceUv(count);
    for(uint32_t i=0;i<count;i++) {
        float values[15];
        uint32_t color;
        input.read(reinterpret_cast<char*>(values),sizeof(values));
        input.read(reinterpret_cast<char*>(&color),sizeof(color));
        if(!input) {
            std::fprintf(stderr,"Truncated Coastal Cliff High cache\n");
            std::exit(1);
        }
        triangles[i]={
            {values[0],values[1],values[2]},
            {values[3],values[4],values[5]},
            {values[6],values[7],values[8]},int(i)};
        sourceUv[i]={
            values[9],values[10],values[11],values[12],values[13],values[14]};
    }
    std::vector<WideNode> nodes;
    BvhBuildReport report;
    std::string error;
    std::fprintf(
        stdout,"Building TinyBVH spatial-split BVH8 for %u triangles "
        "(up to %d per leaf)...\n",count,leafTriangles);
    std::fflush(stdout);
    if(!buildTinyBvh(
           BvhBuilder::TinyBvhSbvh,leafTriangles,&triangles,&nodes,
           &report,&error)) {
        std::fprintf(stderr,"TinyBVH build failed: %s\n",error.c_str());
        std::exit(1);
    }
    std::vector<TexCoords> texCoords(triangles.size());
    for(size_t i=0;i<triangles.size();i++) {
        int source=triangles[i].sourcePrimitive;
        texCoords[i]=source>=0&&source<int(sourceUv.size())?
            sourceUv[source]:TexCoords{};
    }

    int batches=(int(triangles.size())+3)/4;
    std::vector<LocalFrame> frames(batches);
    std::vector<float> coefficientFloats(size_t(batches)*256);
    auto set3=[&](int batch,int row,int column,Vec3 value) {
        float* p=coefficientFloats.data()+size_t(batch)*256+row*16+column;
        p[0]=value.x;p[1]=value.y;p[2]=value.z;
    };
    int exactOnly=0;
    for(int batch=0;batch<batches;batch++) {
        Bounds bounds;
        int end=std::min(int(triangles.size()),batch*4+4);
        for(int i=batch*4;i<end;i++)includeTriangle(&bounds,triangles[i]);
        Vec3 center=mul(add(bounds.lo,bounds.hi),.5f);
        Vec3 extent=sub(bounds.hi,bounds.lo);
        float radius=.5f*std::max({extent.x,extent.y,extent.z});
        frames[batch]={center,1.f/std::max(radius,1e-6f)};
        for(int local=0;local<4;local++) {
            int index=batch*4+local,row=local*4;
            if(index>=int(triangles.size()))continue;
            Triangle q=triangles[index];
            q.a=mul(sub(q.a,center),frames[batch].scale);
            q.c=mul(q.c,frames[batch].scale);
            q.b=mul(q.b,frames[batch].scale);
            Vec3 n=cross(q.c,q.b),pu=cross(q.b,q.a),pv=cross(q.a,q.c);
            coefficientFloats[size_t(batch)*256+(row+0)*16]=dot(q.a,n);
            set3(batch,row+0,1,mul(n,-1));
            set3(batch,row+1,4,pu);
            set3(batch,row+1,7,mul(q.b,-1));
            set3(batch,row+2,4,pv);
            set3(batch,row+2,7,q.c);
            set3(batch,row+3,4,n);
        }
        float* coefficients=coefficientFloats.data()+size_t(batch)*256;
        bool unsafe=!std::isfinite(center.x)||!std::isfinite(center.y)||
                    !std::isfinite(center.z)||
                    !std::isfinite(frames[batch].scale);
        for(int i=0;i<256;i++)
            unsafe|=!std::isfinite(coefficients[i])||
                    std::fabs(coefficients[i])>65504.f;
        if(unsafe) {
            std::fill(coefficients,coefficients+256,0.f);
            frames[batch]={{0,0,0},0};
            exactOnly++;
        }
    }
    std::vector<half> coefficients(coefficientFloats.size());
    for(size_t i=0;i<coefficients.size();i++)
        coefficients[i]=__float2half(coefficientFloats[i]);

    std::vector<uchar4> texture;
    int textureWidth=0,textureHeight=0;
    if(!loadImage(texturePath,&texture,&textureWidth,&textureHeight)) {
        std::fprintf(stderr,"Could not load texture: %s\n",texturePath);
        std::exit(1);
    }

    Scene scene{};
    scene.trianglesCount=int(triangles.size());
    scene.nodeCount=int(nodes.size());
    scene.textureWidth=textureWidth;
    scene.textureHeight=textureHeight;
#define UPLOAD(field,host) do { \
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scene.field), \
                          (host).size()*sizeof((host)[0]))); \
    CUDA_CHECK(cudaMemcpy(scene.field,(host).data(), \
                          (host).size()*sizeof((host)[0]), \
                          cudaMemcpyHostToDevice)); \
} while(0)
    UPLOAD(triangles,triangles);
    UPLOAD(nodes,nodes);
    UPLOAD(coefficients,coefficients);
    UPLOAD(frames,frames);
    UPLOAD(texCoords,texCoords);
    UPLOAD(texture,texture);
#undef UPLOAD
    double elapsed=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    std::fprintf(
        stdout,
        "Ready: source=%u packed=%d nodes=%d leaves=%d texture=%dx%d "
        "exact-only=%d build+upload=%.1f ms\n",
        count,scene.trianglesCount,scene.nodeCount,report.leafCount,
        textureWidth,textureHeight,exactOnly,elapsed);
    return scene;
}

static void freeScene(Scene* scene) {
    cudaFree(scene->triangles);
    cudaFree(scene->nodes);
    cudaFree(scene->coefficients);
    cudaFree(scene->frames);
    cudaFree(scene->texCoords);
    cudaFree(scene->texture);
    *scene={};
}

enum class Backend { Cuda32, Cuda16, Tensor16 };

static const char* backendName(Backend backend) {
    switch(backend) {
    case Backend::Cuda32:return "CUDA32 independent";
    case Backend::Cuda16:return "CUDA-packet16";
    case Backend::Tensor16:return "WMMA-validated (F16->F32 + Moller)";
    }
    return "";
}

static void keyCallback(GLFWwindow* window,int key,int,int action,int) {
    if(action!=GLFW_PRESS)return;
    if(key==GLFW_KEY_ESCAPE)glfwSetWindowShouldClose(window,GLFW_TRUE);
    if(key==GLFW_KEY_SPACE) {
        int backend=glfwGetWindowAttrib(window,GLFW_CONTEXT_VERSION_MINOR);
        (void)backend;
    }
}

int main(int argc,char** argv) {
    const char* geometry=argc>1?argv[1]:
        "build/core/open-model-assets/CoastalCliffHigh.brtri";
    const char* texture=argc>2?argv[2]:
        "build/core/open-model-assets/CoastalCliffHigh-albedo.png";
    int leafTriangles=argc>3?std::max(4,std::atoi(argv[3])):256;
    leafTriangles=(leafTriangles+3)&~3;

    int device=0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
    bool tensorSupported=properties.major>=7;
    std::fprintf(
        stdout,"GPU: %s (SM %d.%d) — WMMA tensor cores %s\n",
        properties.name,properties.major,properties.minor,
        tensorSupported?"supported":"not supported");
    Scene scene=loadScene(geometry,texture,leafTriangles);

    if(!glfwInit()) {
        std::fprintf(stderr,"GLFW initialization failed\n");
        return 1;
    }
    GLFWwindow* window=glfwCreateWindow(
        1280,720,"RayMMA Coastal Cliff High — starting",nullptr,nullptr);
    if(!window) {
        std::fprintf(stderr,"Could not create OpenGL window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window,keyCallback);

    GLuint displayTexture=0;
    glGenTextures(1,&displayTexture);
    glBindTexture(GL_TEXTURE_2D,displayTexture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexImage2D(
        GL_TEXTURE_2D,0,GL_RGBA8,RENDER_W,RENDER_H,0,
        GL_RGBA,GL_UNSIGNED_BYTE,nullptr);

    size_t pixels=size_t(RENDER_W)*RENDER_H;
    Hit* hits=nullptr;
    uchar4* output=nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hits),pixels*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output),pixels*sizeof(uchar4)));
    std::vector<uchar4> hostOutput(pixels);
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    Backend backend=Backend::Cuda32;
    bool paused=false;
    bool previousSpace=false,previousPause=false;
    double previous=glfwGetTime(),angle=0;
    float height=1.15f,smoothedMs=0;
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        bool space=glfwGetKey(window,GLFW_KEY_SPACE)==GLFW_PRESS;
        bool pause=glfwGetKey(window,GLFW_KEY_P)==GLFW_PRESS;
        Backend previousBackend=backend;
        if(space&&!previousSpace) {
            backend=backend==Backend::Cuda32?Backend::Cuda16:
                backend==Backend::Cuda16&&tensorSupported?
                    Backend::Tensor16:Backend::Cuda32;
        }
        if(pause&&!previousPause)paused=!paused;
        previousSpace=space;
        previousPause=pause;
        if(backend!=previousBackend) {
            smoothedMs=0;
            std::fprintf(stdout,"Backend: %s\n",backendName(backend));
        }
        double now=glfwGetTime();
        float delta=float(std::min(now-previous,.1));
        previous=now;
        if(!paused)angle+=delta*.22f;
        if(glfwGetKey(window,GLFW_KEY_LEFT)==GLFW_PRESS)angle-=delta;
        if(glfwGetKey(window,GLFW_KEY_RIGHT)==GLFW_PRESS)angle+=delta;
        if(glfwGetKey(window,GLFW_KEY_UP)==GLFW_PRESS)height+=delta;
        if(glfwGetKey(window,GLFW_KEY_DOWN)==GLFW_PRESS)height-=delta;
        height=std::clamp(height,-.2f,2.8f);
        Camera camera=orbitCamera(float(angle),height);

        CUDA_CHECK(cudaEventRecord(start));
        if(backend==Backend::Cuda32) {
            int blocks=int((pixels+CUDA_THREADS-1)/CUDA_THREADS);
            traceCuda32<<<blocks,CUDA_THREADS>>>(
                hits,RENDER_W,RENDER_H,scene.triangles,scene.nodes,camera);
        } else if(backend==Backend::Cuda16) {
            int blocks=int((pixels+PACKET_RAYS-1)/PACKET_RAYS);
            traceCuda16<<<blocks,32>>>(
                hits,RENDER_W,RENDER_H,scene.triangles,scene.nodes,camera);
        } else {
            int blocks=int((pixels+PACKET_RAYS-1)/PACKET_RAYS);
            traceTensor16<<<blocks,32>>>(
                hits,RENDER_W,RENDER_H,scene.triangles,scene.coefficients,
                scene.frames,scene.nodes,camera);
        }
        int blocks=int((pixels+CUDA_THREADS-1)/CUDA_THREADS);
        shade<<<blocks,CUDA_THREADS>>>(
            output,hits,RENDER_W,RENDER_H,scene.texCoords,scene.texture,
            scene.textureWidth,scene.textureHeight,camera);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float frameMs=0;
        CUDA_CHECK(cudaEventElapsedTime(&frameMs,start,stop));
        smoothedMs=smoothedMs?smoothedMs*.92f+frameMs*.08f:frameMs;
        CUDA_CHECK(cudaMemcpy(
            hostOutput.data(),output,pixels*sizeof(uchar4),
            cudaMemcpyDeviceToHost));

        glBindTexture(GL_TEXTURE_2D,displayTexture);
        glTexSubImage2D(
            GL_TEXTURE_2D,0,0,0,RENDER_W,RENDER_H,
            GL_RGBA,GL_UNSIGNED_BYTE,hostOutput.data());
        int windowWidth=0,windowHeight=0;
        glfwGetFramebufferSize(window,&windowWidth,&windowHeight);
        glViewport(0,0,windowWidth,windowHeight);
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0,1);glVertex2f(-1,-1);
        glTexCoord2f(1,1);glVertex2f( 1,-1);
        glTexCoord2f(1,0);glVertex2f( 1, 1);
        glTexCoord2f(0,0);glVertex2f(-1, 1);
        glEnd();
        char title[512];
        std::snprintf(
            title,sizeof(title),
            "RayMMA Coastal Cliff High | %s | 1920x1080 | %.2f ms GPU (%.1f FPS) "
            "| %d tris/leaf | WMMA %s | Space: cycle  "
            "P: pause  Arrows: orbit",
            backendName(backend),smoothedMs,1000.f/smoothedMs,leafTriangles,
            tensorSupported?"supported":"unsupported");
        glfwSetWindowTitle(window,title);
        glfwSwapBuffers(window);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(hits);
    cudaFree(output);
    glDeleteTextures(1,&displayTexture);
    glfwDestroyWindow(window);
    glfwTerminate();
    freeScene(&scene);
    return 0;
}

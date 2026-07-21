// SPDX-License-Identifier: MIT
//
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "production_bvh.h"

#ifndef RAYMMA_VERSION
#define RAYMMA_VERSION "development"
#endif
#ifndef RAYMMA_BUILD_CONFIG
#define RAYMMA_BUILD_CONFIG "unknown"
#endif

using namespace nvcuda;

constexpr int SAH_BINS=16;
constexpr int DEFAULT_LEAF_TRIANGLES=16;
constexpr int PACKET_RAYS=16;
constexpr int CUDA_PACKET_RAYS=32;
constexpr int CUDA_BLOCK_THREADS=4*CUDA_PACKET_RAYS;
constexpr int MAX_PACKET_LEAVES=1024;
constexpr float APPROXIMATE_DETERMINANT_FLOOR=1e-5f;

enum class TensorVariant {
    Validated,
    UvtDepthSorted,
    E0E1E2,
};

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__, \
                 cudaGetErrorString(e));std::exit(1); } } while(0)

struct Ray { Vec3 o,d; };
struct Hit { float t,u,v;int triangle; };
struct LocalFrame { Vec3 center;float scale; };

struct PacketLeaf {
    int first;
    int count;
    uint32_t rayMask;
};

struct DeviceStats {
    unsigned long long nodeVisits;
    unsigned long long childAabbTests;
    unsigned long long leafPacketVisits;
    unsigned long long rayLeafVisits;
    unsigned long long triangleTests;
    unsigned long long tensorTiles;
    unsigned long long exactTests;
    unsigned long long numericFallbacks;
    unsigned long long stackOverflows;
};

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
    return mul(a,rsqrtf(dot(a,a)));
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
    float det=dot(q.c,p);
    if(fabsf(det)<1e-7f)return false;
    float inv=1.f/det;
    *u=dot(s,p)*inv;
    Vec3 z=cross(s,q.c);
    *v=dot(ray.d,z)*inv;*t=dot(q.b,z)*inv;
    return *u>=0&&*v>=0&&*u+*v<=1&&*t>.001f&&*t<nearest;
}

__global__ void bruteTrace(
    Hit* output,const Ray* rays,int rayCount,const Triangle* triangles,
    int triangleCount) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=rayCount)return;
    Ray ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    for(int triangle=0;triangle<triangleCount;triangle++) {
        float t,u,v;
        if(intersectExact(
               triangles[triangle],ray,hit.t,&t,&u,&v))
            hit={t,u,v,triangle};
    }
    output[pixel]=hit;
}

__device__ void addTraversalStats(
    DeviceStats* stats,uint32_t mask,bool leaf,int triangles) {
    if(!stats||threadIdx.x!=0)return;
    atomicAdd(&stats->childAabbTests,1ull);
    if(mask&&leaf) {
        atomicAdd(&stats->leafPacketVisits,1ull);
        int rays=__popc(mask);
        atomicAdd(&stats->rayLeafVisits,
                  static_cast<unsigned long long>(rays));
        atomicAdd(&stats->triangleTests,
                  static_cast<unsigned long long>(rays)*triangles);
    }
}

__global__ void traceMatchedWide(
    Hit* output,const Ray* rays,int rayCount,const Triangle* triangles,
    const WideNode* nodes,DeviceStats* stats) {
    int lane=threadIdx.x,firstRay=blockIdx.x*PACKET_RAYS;
    int pixel=firstRay+lane;
    __shared__ int stack[64];
    __shared__ int stackTop;
    Ray ray{{0,0,0},{0,0,1}};
    if(lane<PACKET_RAYS&&pixel<rayCount)ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    if(lane==0){stack[0]=0;stackTop=1;}
    __syncwarp();
    while(true) {
        int nodeIndex=-1;
        if(lane==0&&stackTop>0)nodeIndex=stack[--stackTop];
        nodeIndex=__shfl_sync(0xffffffffu,nodeIndex,0);
        if(nodeIndex<0)break;
        WideNode node=nodes[nodeIndex];
        if(stats&&lane==0)atomicAdd(&stats->nodeVisits,1ull);
        for(int slot=0;slot<node.childCount;slot++) {
            bool rayHits=lane<PACKET_RAYS&&pixel<rayCount&&
                intersectAabb(ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t);
            uint32_t mask=__ballot_sync(0xffffffffu,rayHits)&0xffffu;
            bool leaf=node.child[slot]<0;
            addTraversalStats(
                stats,mask,leaf,node.count[slot]);
            if(!mask)continue;
            if(!leaf) {
                if(lane==0) {
                    if(stackTop<64)stack[stackTop++]=node.child[slot];
                    else if(stats)atomicAdd(&stats->stackOverflows,1ull);
                }
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

// Full-warp CUDA control: one independent ray per lane with a private short
// stack and FP32 Moller-Trumbore leaf tests. Independent traversal avoids
// charging CUDA for the union of all nodes touched by an incoherent warp.
__global__ __launch_bounds__(CUDA_BLOCK_THREADS,4) void traceCuda32Wide(
    Hit* __restrict__ output,const Ray* __restrict__ rays,int rayCount,
    const Triangle* __restrict__ triangles,
    const WideNode* __restrict__ nodes,DeviceStats* stats) {
    int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=rayCount)return;
    int stack[32];
    int stackTop=1;
    stack[0]=0;
    Ray ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    while(stackTop) {
        int nodeIndex=stack[--stackTop];
        WideNode node=nodes[nodeIndex];
        if(stats)atomicAdd(&stats->nodeVisits,1ull);
        for(int slot=0;slot<node.childCount;slot++) {
            if(stats)atomicAdd(&stats->childAabbTests,1ull);
            bool rayHits=intersectAabb(
                ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t);
            bool leaf=node.child[slot]<0;
            if(!rayHits)continue;
            if(!leaf) {
                if(stackTop<32)stack[stackTop++]=node.child[slot];
                else if(stats)atomicAdd(&stats->stackOverflows,1ull);
                continue;
            }
            if(stats) {
                atomicAdd(&stats->leafPacketVisits,1ull);
                atomicAdd(&stats->rayLeafVisits,1ull);
                atomicAdd(
                    &stats->triangleTests,
                    static_cast<unsigned long long>(node.count[slot]));
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

__device__ uint32_t hash32(uint32_t value) {
    value^=value>>16;
    value*=0x7feb352du;
    value^=value>>15;
    value*=0x846ca68bu;
    return value^(value>>16);
}

__device__ float hashUnit(uint32_t value) {
    return float(hash32(value)>>8)*(1.f/16777216.f);
}

__global__ void generateDiffuseSecondary(
    Ray* output,int* valid,const Ray* primary,const Hit* hits,int rayCount,
    const Triangle* triangles,const int* originalPixels) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=rayCount)return;
    Hit hit=hits[index];
    if(hit.triangle<0) {
        valid[index]=0;
        output[index]={};
        return;
    }
    Triangle triangle=triangles[hit.triangle];
    Vec3 normal=unit(cross(triangle.c,triangle.b));
    if(dot(normal,primary[index].d)>0)normal=mul(normal,-1.f);
    Vec3 axis=fabsf(normal.z)<.999f?Vec3{0,0,1}:Vec3{0,1,0};
    Vec3 tangent=unit(cross(axis,normal));
    Vec3 bitangent=cross(normal,tangent);
    uint32_t seed=uint32_t(originalPixels[index])^
                  (uint32_t(triangle.sourcePrimitive)*0x9e3779b9u);
    float u1=hashUnit(seed^0x68bc21ebu);
    float u2=hashUnit(seed^0x02e5be93u);
    float radius=sqrtf(u1);
    float phi=6.28318530718f*u2;
    Vec3 direction=add(
        mul(normal,sqrtf(fmaxf(0.f,1.f-u1))),
        add(mul(tangent,radius*cosf(phi)),
            mul(bitangent,radius*sinf(phi))));
    Vec3 position=add(primary[index].o,mul(primary[index].d,hit.t));
    output[index]={
        add(position,mul(normal,4e-4f)),
        unit(direction)};
    valid[index]=1;
}

template<TensorVariant Variant>
__device__ void runTensorLeaf(
    Hit* hit,Ray ray,uint32_t rayMask,int lane,int firstTriangle,
    int triangleCount,const Triangle* triangles,const half* coefficients,
    const LocalFrame* frames,half* features,float* values,
    unsigned long long* broadPassed,
    unsigned long long* numericFallbacks) {
    int firstBatch=firstTriangle/4;
    int endTriangle=firstTriangle+triangleCount;
    int endBatch=(endTriangle+3)/4;
    for(int batch=firstBatch;batch<endBatch;batch++) {
        LocalFrame frame=frames[batch];
        bool unsafeBatch=!(frame.scale>0.f);
        if constexpr(Variant!=TensorVariant::Validated)
            if(unsafeBatch)continue;
        bool unsafeFeatures=unsafeBatch;
        if(lane<PACKET_RAYS) {
            Vec3 origin=unsafeBatch?Vec3{0,0,0}:
                mul(sub(ray.o,frame.center),frame.scale);
            Vec3 m=cross(origin,ray.d);
            float input[10]={
                1,origin.x,origin.y,origin.z,
                ray.d.x,ray.d.y,ray.d.z,m.x,m.y,m.z};
            if constexpr(Variant==TensorVariant::Validated) {
                #pragma unroll
                for(int k=0;k<10;k++)
                    unsafeFeatures|=
                        !isfinite(input[k])||fabsf(input[k])>65504.f;
            }
            half* f=features+lane*16;
            #pragma unroll
            for(int k=0;k<16;k++)
                f[k]=__float2half(
                    k<10&&
                    (Variant!=TensorVariant::Validated||!unsafeFeatures)?
                    input[k]:0.f);
        }
        __syncwarp();
        wmma::fragment<
            wmma::matrix_a,16,16,16,half,wmma::row_major> af;
        wmma::fragment<
            wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
        wmma::fragment<
            wmma::accumulator,16,16,16,float> cf;
        wmma::fill_fragment(cf,0.f);
        wmma::load_matrix_sync(af,coefficients+batch*256,16);
        wmma::load_matrix_sync(bf,features,16);
        wmma::mma_sync(cf,af,bf,cf);
        wmma::store_matrix_sync(values,cf,16,wmma::mem_row_major);
        __syncwarp();
        if(lane<PACKET_RAYS&&(rayMask&(1u<<lane))) {
            #pragma unroll
            for(int local=0;local<4;local++) {
                int triangle=batch*4+local;
                if(triangle<firstTriangle||triangle>=endTriangle)continue;
                if constexpr(Variant==TensorVariant::Validated) {
                    float nu=values[(local*4+1)*16+lane];
                    float nv=values[(local*4+2)*16+lane];
                    float de=values[(local*4+3)*16+lane];
                    bool numericAmbiguous=unsafeFeatures||
                        !isfinite(nu)||!isfinite(nv)||!isfinite(de);
                    bool ambiguous=numericAmbiguous||fabsf(de)<1e-5f;
                    float sign=de<0?-1.f:1.f,d=sign*de;
                    float u=sign*nu,v=sign*nv;
                    float tolerance=2.f*d;
                    // FP16 is a broad phase only. A near-zero approximate
                    // determinant or non-finite FP16 data is numerically
                    // ambiguous, not proof of a miss. Likewise, approximate
                    // t can flip sign under cancellation, so FP32 below
                    // exclusively owns the depth predicate.
                    if(ambiguous||
                       (u>=-tolerance&&v>=-tolerance&&
                        u+v<=d+tolerance)) {
                        (*broadPassed)++;
                        if(numericAmbiguous)(*numericFallbacks)++;
                        float exactT,exactU,exactV;
                        if(intersectExact(
                               triangles[triangle],ray,hit->t,
                               &exactT,&exactU,&exactV))
                            *hit={exactT,exactU,exactV,triangle};
                    }
                } else if constexpr(
                    Variant==TensorVariant::UvtDepthSorted) {
                    float nt=values[(local*4+0)*16+lane];
                    float nu=values[(local*4+1)*16+lane];
                    float nv=values[(local*4+2)*16+lane];
                    float de=values[(local*4+3)*16+lane];
                    if(fabsf(de)>=APPROXIMATE_DETERMINANT_FLOOR) {
                        float sign=de<0?-1.f:1.f,d=sign*de;
                        float u=sign*nu,v=sign*nv;
                        if(u>=0.f&&v>=0.f&&u+v<=d) {
                            float inverse=1.f/de;
                            float worldT=(nt*inverse)/frame.scale;
                            if(worldT>.001f&&worldT<hit->t) {
                                (*broadPassed)++;
                                *hit={worldT,nu*inverse,nv*inverse,triangle};
                            }
                        }
                    }
                } else {
                    float e0=values[(local*4+0)*16+lane];
                    float e1=values[(local*4+1)*16+lane];
                    float e2=values[(local*4+2)*16+lane];
                    float nt=values[(local*4+3)*16+lane];
                    float de=e0+e1+e2;
                    if(fabsf(de)>=APPROXIMATE_DETERMINANT_FLOOR) {
                        float sign=de<0?-1.f:1.f;
                        if(sign*e0>=0.f&&sign*e1>=0.f&&sign*e2>=0.f) {
                            float inverse=1.f/de;
                            float worldT=(nt*inverse)/frame.scale;
                            if(worldT>.001f&&worldT<hit->t) {
                                (*broadPassed)++;
                                *hit={worldT,e1*inverse,e2*inverse,triangle};
                            }
                        }
                    }
                }
            }
        }
        __syncwarp();
    }
}

template<TensorVariant Variant>
__global__ void traceTensorWide(
    Hit* output,const Ray* rays,int rayCount,const Triangle* triangles,
    const half* coefficients,const LocalFrame* frames,
    const WideNode* nodes,DeviceStats* stats) {
    int lane=threadIdx.x,firstRay=blockIdx.x*PACKET_RAYS;
    int pixel=firstRay+lane;
    __shared__ __align__(32) half features[16*16];
    __shared__ __align__(32) float values[16*16];
    __shared__ int stack[64];
    __shared__ int stackTop;
    Ray ray{{0,0,0},{0,0,1}};
    if(lane<PACKET_RAYS&&pixel<rayCount)ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    unsigned long long broadPassed=0,numericFallbacks=0,tensorTiles=0;
    if(lane==0){stack[0]=0;stackTop=1;}
    __syncwarp();
    while(true) {
        int nodeIndex=-1;
        if(lane==0&&stackTop>0)nodeIndex=stack[--stackTop];
        nodeIndex=__shfl_sync(0xffffffffu,nodeIndex,0);
        if(nodeIndex<0)break;
        WideNode node=nodes[nodeIndex];
        if(stats&&lane==0)atomicAdd(&stats->nodeVisits,1ull);
        for(int slot=0;slot<node.childCount;slot++) {
            bool rayHits=lane<PACKET_RAYS&&pixel<rayCount&&
                intersectAabb(ray.o,ray.d,node.lo[slot],node.hi[slot],hit.t);
            uint32_t mask=__ballot_sync(0xffffffffu,rayHits)&0xffffu;
            bool leaf=node.child[slot]<0;
            addTraversalStats(
                stats,mask,leaf,node.count[slot]);
            if(!mask)continue;
            if(!leaf) {
                if(lane==0) {
                    if(stackTop<64)stack[stackTop++]=node.child[slot];
                    else if(stats)atomicAdd(&stats->stackOverflows,1ull);
                }
                continue;
            }
            tensorTiles+=(node.count[slot]+3)/4;
            runTensorLeaf<Variant>(
                &hit,ray,mask,lane,node.first[slot],node.count[slot],
                triangles,coefficients,frames,features,values,&broadPassed,
                &numericFallbacks);
        }
    }
    if(stats&&lane<PACKET_RAYS) {
        if(lane==0)atomicAdd(&stats->tensorTiles,tensorTiles);
        atomicAdd(&stats->exactTests,broadPassed);
        atomicAdd(&stats->numericFallbacks,numericFallbacks);
    }
    if(lane<PACKET_RAYS&&pixel<rayCount)output[pixel]=hit;
}

__global__ void collectPacketLeaves(
    const Ray* rays,int rayCount,const WideNode* nodes,PacketLeaf* leaves,
    int* leafCounts,int* overflow,DeviceStats* stats) {
    int lane=threadIdx.x,packet=blockIdx.x,pixel=packet*PACKET_RAYS+lane;
    __shared__ int stack[64];
    __shared__ int stackTop;
    __shared__ int leafTop;
    Ray ray{{0,0,0},{0,0,1}};
    if(lane<PACKET_RAYS&&pixel<rayCount)ray=rays[pixel];
    if(lane==0){stack[0]=0;stackTop=1;leafTop=0;}
    __syncwarp();
    while(true) {
        int nodeIndex=-1;
        if(lane==0&&stackTop>0)nodeIndex=stack[--stackTop];
        nodeIndex=__shfl_sync(0xffffffffu,nodeIndex,0);
        if(nodeIndex<0)break;
        WideNode node=nodes[nodeIndex];
        if(stats&&lane==0)atomicAdd(&stats->nodeVisits,1ull);
        for(int slot=0;slot<node.childCount;slot++) {
            bool rayHits=lane<PACKET_RAYS&&pixel<rayCount&&
                intersectAabb(
                    ray.o,ray.d,node.lo[slot],node.hi[slot],1e30f);
            uint32_t mask=__ballot_sync(0xffffffffu,rayHits)&0xffffu;
            bool leaf=node.child[slot]<0;
            addTraversalStats(
                stats,mask,leaf,node.count[slot]);
            if(!mask)continue;
            if(!leaf) {
                if(lane==0) {
                    if(stackTop<64)stack[stackTop++]=node.child[slot];
                    else if(stats)atomicAdd(&stats->stackOverflows,1ull);
                }
            } else if(lane==0) {
                int index=leafTop++;
                if(index<MAX_PACKET_LEAVES)
                    leaves[packet*MAX_PACKET_LEAVES+index]={
                        node.first[slot],node.count[slot],mask};
                else atomicAdd(overflow,1);
            }
        }
        __syncwarp();
    }
    if(lane==0)leafCounts[packet]=min(leafTop,MAX_PACKET_LEAVES);
}

__global__ void traceMatchedLeaves(
    Hit* output,const Ray* rays,int rayCount,const Triangle* triangles,
    const PacketLeaf* leaves,const int* leafCounts) {
    int lane=threadIdx.x,packet=blockIdx.x,pixel=packet*PACKET_RAYS+lane;
    Ray ray{{0,0,0},{0,0,1}};
    if(lane<PACKET_RAYS&&pixel<rayCount)ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    int count=leafCounts[packet];
    for(int leaf=0;leaf<count;leaf++) {
        PacketLeaf task=leaves[packet*MAX_PACKET_LEAVES+leaf];
        if(lane<PACKET_RAYS&&(task.rayMask&(1u<<lane))) {
            int end=task.first+task.count;
            for(int triangle=task.first;triangle<end;triangle++) {
                float t,u,v;
                if(intersectExact(
                       triangles[triangle],ray,hit.t,&t,&u,&v))
                    hit={t,u,v,triangle};
            }
        }
        __syncwarp();
    }
    if(lane<PACKET_RAYS&&pixel<rayCount)output[pixel]=hit;
}

template<TensorVariant Variant>
__global__ void traceTensorLeaves(
    Hit* output,const Ray* rays,int rayCount,const Triangle* triangles,
    const half* coefficients,const LocalFrame* frames,
    const PacketLeaf* leaves,const int* leafCounts) {
    int lane=threadIdx.x,packet=blockIdx.x,pixel=packet*PACKET_RAYS+lane;
    __shared__ __align__(32) half features[16*16];
    __shared__ __align__(32) float values[16*16];
    Ray ray{{0,0,0},{0,0,1}};
    if(lane<PACKET_RAYS&&pixel<rayCount)ray=rays[pixel];
    Hit hit{1e30f,0,0,-1};
    unsigned long long ignored=0,ignoredFallbacks=0;
    int count=leafCounts[packet];
    for(int leaf=0;leaf<count;leaf++) {
        PacketLeaf task=leaves[packet*MAX_PACKET_LEAVES+leaf];
        runTensorLeaf<Variant>(
            &hit,ray,task.rayMask,lane,task.first,task.count,triangles,
            coefficients,frames,features,values,&ignored,&ignoredFallbacks);
    }
    if(lane<PACKET_RAYS&&pixel<rayCount)output[pixel]=hit;
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
    bool valid=false;

    void include(Vec3 p) {
        lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
        hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
        valid=true;
    }
    void include(const Bounds& b) {
        if(!b.valid)return;
        include(b.lo);include(b.hi);
    }
    float area() const {
        if(!valid)return 0;
        Vec3 e=sub(hi,lo);
        return 2.f*(e.x*e.y+e.y*e.z+e.z*e.x);
    }
};

static Vec3 triangleCentroid(const Triangle& q) {
    return mul(add(add(q.a,add(q.a,q.c)),add(q.a,q.b)),1.f/3.f);
}

static Bounds triangleBounds(const Triangle& q) {
    Bounds b;b.include(q.a);b.include(add(q.a,q.c));b.include(add(q.a,q.b));
    return b;
}

static float axisValue(Vec3 p,int axis) {
    return axis==0?p.x:axis==1?p.y:p.z;
}

struct BuildNode {
    Bounds bounds;
    int first=0,count=0,left=-1,right=-1;
    bool leaf() const{return left<0;}
};

struct SahBuilder {
    std::vector<Triangle>& triangles;
    std::vector<BuildNode> nodes;
    int leafTriangles;

    int build(int first,int count) {
        BuildNode node{};node.first=first;node.count=count;
        Bounds centroidBounds;
        for(int i=first;i<first+count;i++) {
            node.bounds.include(triangleBounds(triangles[i]));
            centroidBounds.include(triangleCentroid(triangles[i]));
        }
        int index=int(nodes.size());nodes.push_back(node);
        if(count<=leafTriangles)return index;

        int bestAxis=-1,bestLeft=0;
        float bestCost=std::numeric_limits<float>::max();
        struct Bin { Bounds bounds;int count=0; };
        for(int axis=0;axis<3;axis++) {
            float lower=axisValue(centroidBounds.lo,axis);
            float extent=axisValue(centroidBounds.hi,axis)-lower;
            if(extent<1e-12f)continue;
            std::array<Bin,SAH_BINS> bins{};
            for(int i=first;i<first+count;i++) {
                int bin=std::min(
                    SAH_BINS-1,int((axisValue(
                        triangleCentroid(triangles[i]),axis)-lower)/
                        extent*SAH_BINS));
                bins[bin].count++;
                bins[bin].bounds.include(triangleBounds(triangles[i]));
            }
            std::array<Bounds,SAH_BINS> leftBounds,rightBounds;
            std::array<int,SAH_BINS> leftCounts{},rightCounts{};
            Bounds left,right;int leftCount=0,rightCount=0;
            for(int i=0;i<SAH_BINS;i++) {
                left.include(bins[i].bounds);leftCount+=bins[i].count;
                leftBounds[i]=left;leftCounts[i]=leftCount;
                int r=SAH_BINS-1-i;
                right.include(bins[r].bounds);rightCount+=bins[r].count;
                rightBounds[r]=right;rightCounts[r]=rightCount;
            }
            for(int boundary=0;boundary<SAH_BINS-1;boundary++) {
                int lc=leftCounts[boundary],rc=rightCounts[boundary+1];
                if(!lc||!rc)continue;
                float cost=leftBounds[boundary].area()*lc+
                           rightBounds[boundary+1].area()*rc;
                if(cost<bestCost) {
                    bestCost=cost;bestAxis=axis;
                    bestLeft=lc;
                }
            }
        }
        if(bestAxis<0) {
            Vec3 e=sub(centroidBounds.hi,centroidBounds.lo);
            bestAxis=e.y>e.x&&e.y>=e.z?1:(e.z>e.x?2:0);
            bestLeft=count/2;
        }
        int leftCount=((bestLeft+2)/4)*4;
        int maxLeft=std::max(4,((count-4)/4)*4);
        leftCount=std::clamp(leftCount,4,maxLeft);
        int middle=first+leftCount;
        std::nth_element(
            triangles.begin()+first,triangles.begin()+middle,
            triangles.begin()+first+count,[&](const Triangle& a,const Triangle& b) {
                return axisValue(triangleCentroid(a),bestAxis)<
                       axisValue(triangleCentroid(b),bestAxis);
            });
        int left=build(first,leftCount);
        int right=build(middle,count-leftCount);
        nodes[index].left=left;nodes[index].right=right;
        return index;
    }
};

static int emitWideNode(
    int binaryIndex,const std::vector<BuildNode>& binary,
    std::vector<WideNode>* wide) {
    int result=int(wide->size());wide->emplace_back();
    std::vector<int> frontier{binaryIndex};
    while(frontier.size()<BVH_WIDTH) {
        int expand=-1;float score=-1;
        for(int i=0;i<int(frontier.size());i++) {
            const BuildNode& node=binary[frontier[i]];
            if(!node.leaf()) {
                float candidate=node.bounds.area()*node.count;
                if(candidate>score){score=candidate;expand=i;}
            }
        }
        if(expand<0)break;
        int node=frontier[expand];
        frontier[expand]=binary[node].left;
        frontier.insert(frontier.begin()+expand+1,binary[node].right);
    }
    WideNode output{};output.childCount=int(frontier.size());
    for(int slot=0;slot<output.childCount;slot++) {
        const BuildNode& node=binary[frontier[slot]];
        output.lo[slot]=node.bounds.lo;output.hi[slot]=node.bounds.hi;
        if(node.leaf()) {
            output.child[slot]=-1;output.first[slot]=node.first;
            output.count[slot]=node.count;
        } else {
            output.child[slot]=emitWideNode(frontier[slot],binary,wide);
            output.first[slot]=0;output.count[slot]=0;
        }
    }
    (*wide)[result]=output;
    return result;
}

struct CameraSpec { Vec3 origin,target,up; };
struct HostScene {
    std::string name;
    std::vector<Triangle> triangles;
    CameraSpec camera;
};

static int parseObjIndex(const std::string& token,int vertexCount) {
    size_t slash=token.find('/');
    int value=std::stoi(token.substr(0,slash));
    return value<0?vertexCount+value:value-1;
}

static HostScene loadObj(
    const char* path,const char* name,CameraSpec camera) {
    auto begin=std::chrono::steady_clock::now();
    std::ifstream input(path);
    if(!input)return std::fprintf(stderr,"Cannot open %s\n",path),
                     std::exit(1),HostScene{};
    std::vector<Vec3> vertices;
    std::vector<std::array<int,3>> indices;
    Bounds bounds;
    std::string line;
    while(std::getline(input,line)) {
        if(line.size()>2&&line[0]=='v'&&line[1]==' ') {
            std::istringstream stream(line.substr(2));
            Vec3 v{};stream>>v.x>>v.y>>v.z;
            vertices.push_back(v);bounds.include(v);
        } else if(line.size()>2&&line[0]=='f'&&line[1]==' ') {
            std::istringstream stream(line.substr(2));
            std::vector<int> face;std::string token;
            while(stream>>token)
                face.push_back(parseObjIndex(token,int(vertices.size())));
            for(size_t i=1;i+1<face.size();i++)
                indices.push_back({face[0],face[i],face[i+1]});
        }
    }
    Vec3 center=mul(add(bounds.lo,bounds.hi),.5f);
    Vec3 extent=sub(bounds.hi,bounds.lo);
    float scale=2.f/std::max({extent.x,extent.y,extent.z});
    auto transform=[&](Vec3 p){return mul(sub(p,center),scale);};
    HostScene scene{};scene.name=name;scene.triangles.reserve(indices.size());
    for(auto index:indices) {
        Vec3 a=transform(vertices[index[0]]);
        Vec3 c=transform(vertices[index[1]]);
        Vec3 b=transform(vertices[index[2]]);
        scene.triangles.push_back({a,sub(c,a),sub(b,a)});
    }
    scene.camera={
        transform(camera.origin),transform(camera.target),camera.up};
    double ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    std::printf("loaded %-10s %9zu triangles in %.1f ms\n",
                name,scene.triangles.size(),ms);
    return scene;
}

static HostScene loadBrtri(
    const char* path,const char* name,CameraSpec camera) {
    struct CachedTriangle {
        Vec3 a,c,b;
        float uv[6];
        uint32_t albedo;
    };
    std::ifstream input(path,std::ios::binary);
    char magic[8]{};uint32_t count=0;
    input.read(magic,8);input.read(reinterpret_cast<char*>(&count),4);
    if(!input||std::memcmp(magic,"BRTRI003",8))
        return std::fprintf(stderr,"Invalid cache %s\n",path),
               std::exit(1),HostScene{};
    std::vector<CachedTriangle> cached(count);
    input.read(
        reinterpret_cast<char*>(cached.data()),
        cached.size()*sizeof(CachedTriangle));
    if(!input)return std::fprintf(stderr,"Truncated cache %s\n",path),
                     std::exit(1),HostScene{};
    HostScene scene{};scene.name=name;scene.camera=camera;
    scene.triangles.reserve(count);
    for(const auto& q:cached)scene.triangles.push_back({q.a,q.c,q.b});
    std::printf("loaded %-10s %9zu triangles\n",name,scene.triangles.size());
    return scene;
}

enum class RayMode { Primary,Secondary };

static const char* rayModeName(RayMode mode) {
    return mode==RayMode::Primary?"primary":"secondary-diffuse";
}

static const char* tensorVariantName(TensorVariant variant) {
    switch(variant) {
        case TensorVariant::Validated:return "validated";
        case TensorVariant::UvtDepthSorted:return "uvt-depthsorted";
        case TensorVariant::E0E1E2:return "e0e1e2";
    }
    return "unknown";
}

static bool tensorVariantIsApproximate(TensorVariant variant) {
    return variant!=TensorVariant::Validated;
}

static std::vector<Ray> makePrimaryRays(
    int w,int h,CameraSpec camera,std::vector<int>* originalPixels) {
    Vec3 forward=unit(sub(camera.target,camera.origin));
    Vec3 right=unit(cross(forward,camera.up));
    Vec3 up=unit(cross(right,forward));
    std::vector<Ray> rays(w*h);
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        float px=(2.f*(x+.5f)/w-1.f)*(float(w)/h)*.72f;
        float py=(1.f-2.f*(y+.5f)/h)*.72f;
        rays[y*w+x]={camera.origin,
                     unit(add(forward,add(mul(right,px),mul(up,py))))};
    }
    originalPixels->resize(rays.size());
    std::iota(originalPixels->begin(),originalPixels->end(),0);
    return rays;
}

static void shuffleRays(
    std::vector<Ray>* rays,std::vector<int>* originalPixels) {
    std::vector<int> order(rays->size());
    std::iota(order.begin(),order.end(),0);
    std::mt19937 generator(0x51a7c0de);
    std::shuffle(order.begin(),order.end(),generator);
    std::vector<Ray> shuffledRays(rays->size());
    std::vector<int> shuffledPixels(rays->size());
    for(size_t i=0;i<order.size();i++) {
        shuffledRays[i]=(*rays)[order[i]];
        shuffledPixels[i]=(*originalPixels)[order[i]];
    }
    *rays=std::move(shuffledRays);
    *originalPixels=std::move(shuffledPixels);
}

static uint64_t raySetHash(
    const std::vector<Ray>& rays,const std::vector<int>& originalPixels) {
    uint64_t hash=1469598103934665603ull;
    auto append=[&](uint32_t value) {
        hash^=value;
        hash*=1099511628211ull;
    };
    for(size_t i=0;i<rays.size();i++) {
        append(uint32_t(originalPixels[i]));
        const float values[]={
            rays[i].o.x,rays[i].o.y,rays[i].o.z,
            rays[i].d.x,rays[i].d.y,rays[i].d.z};
        for(float value:values) {
            uint32_t bits;
            std::memcpy(&bits,&value,sizeof(bits));
            append(bits);
        }
    }
    return hash;
}

struct DeviceScene {
    Triangle* triangles=nullptr;
    half* coefficients=nullptr;
    LocalFrame* frames=nullptr;
    WideNode* nodes=nullptr;
    int triangleCount=0,sourceTriangleCount=0,nodeCount=0,exactOnlyBatches=0;
};

#if !RAYMMA_HAS_TINYBVH
const char* bvhBuilderName(BvhBuilder builder) {
    return builder==BvhBuilder::BuiltinSah?
        "builtin-binned-SAH":"TinyBVH-unavailable";
}
bool tinyBvhAvailable() { return false; }
bool buildTinyBvh(
    BvhBuilder,int,std::vector<Triangle>*,std::vector<WideNode>*,
    BvhBuildReport*,std::string* error) {
    *error="this binary was built without TinyBVH";
    return false;
}
#endif

static DeviceScene uploadScene(
    std::vector<Triangle> triangles,std::vector<WideNode>* hostWide,
    double* buildMs,int leafTriangles,BvhBuilder builderChoice,
    TensorVariant tensorVariant,BvhBuildReport* buildReport) {
    auto begin=std::chrono::steady_clock::now();
    int sourceTriangleCount=int(triangles.size());
    for(int i=0;i<sourceTriangleCount;i++)
        if(triangles[i].sourcePrimitive<0)
            triangles[i].sourcePrimitive=i;
    if(builderChoice==BvhBuilder::BuiltinSah) {
        SahBuilder builder{triangles,{},leafTriangles};
        builder.nodes.reserve(triangles.size()/8);
        int root=builder.build(0,int(triangles.size()));
        *buildReport={};
        buildReport->name=bvhBuilderName(builderChoice);
        buildReport->sourceTriangles=sourceTriangleCount;
        buildReport->packedTriangles=int(triangles.size());
        buildReport->minLeafTriangles=std::numeric_limits<int>::max();
        for(const BuildNode& node:builder.nodes) {
            if(!node.leaf())continue;
            buildReport->leafCount++;
            buildReport->minLeafTriangles=
                std::min(buildReport->minLeafTriangles,node.count);
            buildReport->maxLeafTriangles=
                std::max(buildReport->maxLeafTriangles,node.count);
            buildReport->meanLeafTriangles+=node.count;
        }
        buildReport->meanLeafTriangles/=buildReport->leafCount;
        hostWide->reserve(builder.nodes.size()/4);
        emitWideNode(root,builder.nodes,hostWide);
    } else {
        std::string error;
        if(!buildTinyBvh(
               builderChoice,leafTriangles,&triangles,hostWide,
               buildReport,&error)) {
            std::fprintf(stderr,"TinyBVH build failed: %s\n",error.c_str());
            std::exit(1);
        }
    }
    for(const WideNode& node:*hostWide)
        for(int slot=0;slot<node.childCount;slot++)
            if(node.child[slot]<0&&
               (node.first[slot]%4||node.count[slot]<=0||
                node.first[slot]+node.count[slot]>int(triangles.size()))) {
                std::fprintf(
                    stderr,
                    "Invalid WMMA leaf layout: first=%d count=%d "
                    "triangles=%zu\n",
                    node.first[slot],node.count[slot],triangles.size());
                std::exit(1);
            }

    int batches=(int(triangles.size())+3)/4;
    std::vector<LocalFrame> frames(batches);
    std::vector<float> coefficients(size_t(batches)*256);
    int exactOnlyBatches=0;
    auto set3=[&](int batch,int row,int col,Vec3 v) {
        float* p=coefficients.data()+size_t(batch)*256+row*16+col;
        p[0]=v.x;p[1]=v.y;p[2]=v.z;
    };
    for(int batch=0;batch<batches;batch++) {
        Bounds bounds;
        int end=std::min(int(triangles.size()),batch*4+4);
        for(int i=batch*4;i<end;i++)bounds.include(triangleBounds(triangles[i]));
        Vec3 center=mul(add(bounds.lo,bounds.hi),.5f);
        Vec3 extent=sub(bounds.hi,bounds.lo);
        float radius=.5f*std::max({extent.x,extent.y,extent.z});
        frames[batch]={center,1.f/std::max(radius,1e-6f)};
        for(int local=0;local<4;local++) {
            int i=batch*4+local,row=local*4;
            if(i>=int(triangles.size()))continue;
            Triangle q=triangles[i];
            q.a=mul(sub(q.a,center),frames[batch].scale);
            q.c=mul(q.c,frames[batch].scale);
            q.b=mul(q.b,frames[batch].scale);
            Vec3 n=cross(q.c,q.b),pu=cross(q.b,q.a),pv=cross(q.a,q.c);
            if(tensorVariant==TensorVariant::E0E1E2) {
                Vec3 v0=q.a,v1=add(q.a,q.c),v2=add(q.a,q.b);
                // Direct oriented edge rows. Their sum is Delta, while rows
                // one and two are the u and v barycentric numerators.
                set3(batch,row+0,4,cross(v1,v2));
                set3(batch,row+0,7,sub(v2,v1));
                set3(batch,row+1,4,cross(v2,v0));
                set3(batch,row+1,7,sub(v0,v2));
                set3(batch,row+2,4,cross(v0,v1));
                set3(batch,row+2,7,sub(v1,v0));
                coefficients[size_t(batch)*256+(row+3)*16]=dot(q.a,n);
                set3(batch,row+3,1,mul(n,-1));
            } else {
                coefficients[size_t(batch)*256+(row+0)*16]=dot(q.a,n);
                set3(batch,row+0,1,mul(n,-1));
                set3(batch,row+1,4,pu);
                set3(batch,row+1,7,mul(q.b,-1));
                set3(batch,row+2,4,pv);set3(batch,row+2,7,q.c);
                set3(batch,row+3,4,n);
            }
            // A common power-of-two row scale preserves the exact row ratios
            // before FP16 conversion and reduces subnormal/underflow loss for
            // small triangles. It needs no per-hit recovery.
            float* triangleRows=
                coefficients.data()+size_t(batch)*256+row*16;
            float maximumCoefficient=0.f;
            for(int k=0;k<4*16;k++)
                maximumCoefficient=std::max(
                    maximumCoefficient,std::fabs(triangleRows[k]));
            if(maximumCoefficient>0.f&&std::isfinite(maximumCoefficient)) {
                int exponent=0;
                std::frexp(maximumCoefficient,&exponent);
                int shift=std::clamp(-exponent,-120,120);
                float rowScale=std::ldexp(1.f,shift);
                for(int k=0;k<4*16;k++)triangleRows[k]*=rowScale;
            }
        }
        float* batchCoefficients=coefficients.data()+size_t(batch)*256;
        bool unsafe=!std::isfinite(center.x)||!std::isfinite(center.y)||
                    !std::isfinite(center.z)||
                    !std::isfinite(frames[batch].scale);
        for(int i=0;i<256;i++)
            unsafe|=!std::isfinite(batchCoefficients[i])||
                    std::fabs(batchCoefficients[i])>65504.f;
        if(unsafe) {
            std::fill(batchCoefficients,batchCoefficients+256,0.f);
            // Zero scale marks an unrepresentable coefficient batch. The
            // validated mode routes it to FP32; no-Moller modes reject it.
            frames[batch]={{0,0,0},0};
            exactOnlyBatches++;
        }
    }
    std::vector<half> packed(coefficients.size());
    for(size_t i=0;i<packed.size();i++)
        packed[i]=__float2half(coefficients[i]);

    DeviceScene scene{};
    scene.triangleCount=int(triangles.size());
    scene.sourceTriangleCount=sourceTriangleCount;
    scene.nodeCount=int(hostWide->size());
    scene.exactOnlyBatches=exactOnlyBatches;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&scene.triangles),
        triangles.size()*sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(
        scene.triangles,triangles.data(),triangles.size()*sizeof(Triangle),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&scene.coefficients),
        packed.size()*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(
        scene.coefficients,packed.data(),packed.size()*sizeof(half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&scene.frames),
        frames.size()*sizeof(LocalFrame)));
    CUDA_CHECK(cudaMemcpy(
        scene.frames,frames.data(),frames.size()*sizeof(LocalFrame),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&scene.nodes),
        hostWide->size()*sizeof(WideNode)));
    CUDA_CHECK(cudaMemcpy(
        scene.nodes,hostWide->data(),hostWide->size()*sizeof(WideNode),
        cudaMemcpyHostToDevice));
    *buildMs=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    return scene;
}

static void freeScene(DeviceScene* scene) {
    cudaFree(scene->triangles);cudaFree(scene->coefficients);
    cudaFree(scene->frames);cudaFree(scene->nodes);*scene={};
}

static std::vector<Ray> makeSecondaryRays(
    const DeviceScene& scene,const std::vector<Ray>& primary,
    const std::vector<int>& primaryPixels,std::vector<int>* secondaryPixels,
    double* setupMs) {
    auto begin=std::chrono::steady_clock::now();
    std::vector<int> sourcePixels=primaryPixels;
    int rayCount=int(primary.size());
    Ray *devicePrimary,*deviceSecondary;
    Hit* primaryHits;
    int *devicePixels,*deviceValid;
    CUDA_CHECK(cudaMalloc((void**)&devicePrimary,rayCount*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc((void**)&deviceSecondary,rayCount*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc((void**)&primaryHits,rayCount*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&devicePixels,rayCount*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&deviceValid,rayCount*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        devicePrimary,primary.data(),rayCount*sizeof(Ray),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        devicePixels,sourcePixels.data(),rayCount*sizeof(int),
        cudaMemcpyHostToDevice));
    int cudaBlocks=(rayCount+CUDA_BLOCK_THREADS-1)/CUDA_BLOCK_THREADS;
    traceCuda32Wide<<<cudaBlocks,CUDA_BLOCK_THREADS>>>(
        primaryHits,devicePrimary,rayCount,scene.triangles,scene.nodes,nullptr);
    generateDiffuseSecondary<<<(rayCount+255)/256,256>>>(
        deviceSecondary,deviceValid,devicePrimary,primaryHits,rayCount,
        scene.triangles,devicePixels);
    std::vector<Ray> generated(rayCount);
    std::vector<int> valid(rayCount);
    CUDA_CHECK(cudaMemcpy(
        generated.data(),deviceSecondary,rayCount*sizeof(Ray),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        valid.data(),deviceValid,rayCount*sizeof(int),
        cudaMemcpyDeviceToHost));
    cudaFree(devicePrimary);cudaFree(deviceSecondary);cudaFree(primaryHits);
    cudaFree(devicePixels);cudaFree(deviceValid);

    std::vector<Ray> compact;
    compact.reserve(rayCount);
    secondaryPixels->clear();
    secondaryPixels->reserve(rayCount);
    for(int i=0;i<rayCount;i++)if(valid[i]) {
        compact.push_back(generated[i]);
        secondaryPixels->push_back(sourcePixels[i]);
    }
    *setupMs=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    return compact;
}

struct Timing {
    float median=0,p10=0,p90=0;
    std::vector<float> samples;
};

static Timing summarize(std::vector<float> values) {
    Timing result{};
    result.samples=values;
    std::sort(values.begin(),values.end());
    auto at=[&](float q){return values[size_t(q*(values.size()-1))];};
    result.median=at(.5f);result.p10=at(.1f);result.p90=at(.9f);
    return result;
}

static float timeLaunch(const std::function<void()>& launch,int repeats=1) {
    cudaEvent_t begin,end;
    CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin));
    for(int i=0;i<repeats;i++)launch();
    CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));
    float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
    return ms/repeats;
}

static Timing sampleLaunch(
    const std::function<void()>& launch,int samples=9,int repeats=1) {
    std::vector<float> values;values.reserve(samples);
    for(int i=0;i<samples;i++)values.push_back(timeLaunch(launch,repeats));
    return summarize(values);
}

struct Accuracy {
    int compared=0,hitMiss=0,falsePositive=0,falseNegative=0;
    int primitive=0,invalid=0;
    float maxAbsT=0,maxRelT=0;
};

static Accuracy compareHits(
    const std::vector<Hit>& reference,const std::vector<Hit>& candidate,
    int triangleCount=-1) {
    Accuracy result{};result.compared=int(reference.size());
    for(size_t i=0;i<reference.size();i++) {
        if(candidate[i].triangle < -1||
           (triangleCount>=0&&candidate[i].triangle>=triangleCount)) {
            result.invalid++;
            continue;
        }
        bool a=reference[i].triangle>=0,b=candidate[i].triangle>=0;
        bool invalidValues=b&&
            (!std::isfinite(candidate[i].t)||
             !std::isfinite(candidate[i].u)||
             !std::isfinite(candidate[i].v));
        if(invalidValues)result.invalid++;
        if(a!=b) {
            result.hitMiss++;
            if(a)result.falseNegative++;
            else result.falsePositive++;
            continue;
        }
        if(!a)continue;
        if(reference[i].triangle!=candidate[i].triangle)result.primitive++;
        if(invalidValues)continue;
        float absolute=fabsf(reference[i].t-candidate[i].t);
        float relative=absolute/std::max(fabsf(reference[i].t),1e-20f);
        result.maxAbsT=std::max(result.maxAbsT,absolute);
        result.maxRelT=std::max(result.maxRelT,relative);
    }
    return result;
}

struct BenchResult {
    Timing cuda32,matched,tensor,traversal,matchedLeaves,tensorLeaves;
    Accuracy cuda32Accuracy,tensorAccuracy,phasedAccuracy,bruteAccuracy;
    DeviceStats cuda32Stats{},matchedStats{},tensorStats{},traversalStats{};
    std::vector<Hit> cuda32Hits,matchedHits,tensorHits;
    int overflow=0,maxLeaves=0,hitCount=0;
};

static BenchResult benchmarkMode(
    const DeviceScene& scene,const std::vector<Ray>& rays,
    const std::vector<int>& originalPixels,int w,int h,int samples,
    TensorVariant tensorVariant) {
    int rayCount=int(rays.size());
    int packets=(rayCount+PACKET_RAYS-1)/PACKET_RAYS;
    int cudaBlocks=(rayCount+CUDA_BLOCK_THREADS-1)/CUDA_BLOCK_THREADS;
    Ray *deviceRays,*bruteRays;
    Hit *cuda32,*matched,*tensor,*phased,*brute;
    PacketLeaf* leaves;int *leafCounts,*overflow;
    DeviceStats *cuda32Stats,*matchedStats,*tensorStats,*traversalStats;
    CUDA_CHECK(cudaMalloc((void**)&deviceRays,rayCount*sizeof(Ray)));
    CUDA_CHECK(cudaMemcpy(
        deviceRays,rays.data(),rayCount*sizeof(Ray),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&cuda32,rayCount*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&matched,rayCount*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&tensor,rayCount*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&phased,rayCount*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&brute,256*sizeof(Hit)));
    CUDA_CHECK(cudaMalloc((void**)&bruteRays,256*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(
        (void**)&leaves,size_t(packets)*MAX_PACKET_LEAVES*sizeof(PacketLeaf)));
    CUDA_CHECK(cudaMalloc((void**)&leafCounts,packets*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&overflow,sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&cuda32Stats,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMalloc((void**)&matchedStats,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMalloc((void**)&tensorStats,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMalloc((void**)&traversalStats,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMemset(cuda32Stats,0,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMemset(matchedStats,0,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMemset(tensorStats,0,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMemset(traversalStats,0,sizeof(DeviceStats)));
    CUDA_CHECK(cudaMemset(overflow,0,sizeof(int)));

    auto launchCuda32=[&] {
        traceCuda32Wide<<<cudaBlocks,CUDA_BLOCK_THREADS>>>(
            cuda32,deviceRays,rayCount,scene.triangles,scene.nodes,nullptr);
    };
    auto launchMatched=[&] {
        traceMatchedWide<<<packets,32>>>(
            matched,deviceRays,rayCount,scene.triangles,scene.nodes,nullptr);
    };
    auto launchTensor=[&] {
        switch(tensorVariant) {
            case TensorVariant::Validated:
                traceTensorWide<TensorVariant::Validated><<<packets,32>>>(
                    tensor,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,scene.nodes,nullptr);
                break;
            case TensorVariant::UvtDepthSorted:
                traceTensorWide<
                    TensorVariant::UvtDepthSorted><<<packets,32>>>(
                    tensor,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,scene.nodes,nullptr);
                break;
            case TensorVariant::E0E1E2:
                traceTensorWide<TensorVariant::E0E1E2><<<packets,32>>>(
                    tensor,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,scene.nodes,nullptr);
                break;
        }
    };
    auto launchTraversal=[&] {
        collectPacketLeaves<<<packets,32>>>(
            deviceRays,rayCount,scene.nodes,leaves,leafCounts,overflow,nullptr);
    };
    auto launchMatchedLeaves=[&] {
        traceMatchedLeaves<<<packets,32>>>(
            phased,deviceRays,rayCount,scene.triangles,leaves,leafCounts);
    };
    auto launchTensorLeaves=[&] {
        switch(tensorVariant) {
            case TensorVariant::Validated:
                traceTensorLeaves<
                    TensorVariant::Validated><<<packets,32>>>(
                    phased,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,leaves,leafCounts);
                break;
            case TensorVariant::UvtDepthSorted:
                traceTensorLeaves<
                    TensorVariant::UvtDepthSorted><<<packets,32>>>(
                    phased,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,leaves,leafCounts);
                break;
            case TensorVariant::E0E1E2:
                traceTensorLeaves<TensorVariant::E0E1E2><<<packets,32>>>(
                    phased,deviceRays,rayCount,scene.triangles,
                    scene.coefficients,scene.frames,leaves,leafCounts);
                break;
        }
    };
    for(int i=0;i<6;i++) {
        launchTraversal();launchCuda32();launchMatched();launchTensor();
        launchMatchedLeaves();launchTensorLeaves();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    BenchResult result{};
    std::vector<float> cuda32Times,matchedTimes,tensorTimes;
    for(int sample=0;sample<samples;sample++) {
        if(sample%3==0) {
            cuda32Times.push_back(timeLaunch(launchCuda32));
            matchedTimes.push_back(timeLaunch(launchMatched));
            tensorTimes.push_back(timeLaunch(launchTensor));
        } else if(sample%3==1) {
            matchedTimes.push_back(timeLaunch(launchMatched));
            tensorTimes.push_back(timeLaunch(launchTensor));
            cuda32Times.push_back(timeLaunch(launchCuda32));
        } else {
            tensorTimes.push_back(timeLaunch(launchTensor));
            cuda32Times.push_back(timeLaunch(launchCuda32));
            matchedTimes.push_back(timeLaunch(launchMatched));
        }
    }
    result.cuda32=summarize(cuda32Times);
    result.matched=summarize(matchedTimes);
    result.tensor=summarize(tensorTimes);
    result.traversal=sampleLaunch(launchTraversal,samples);
    result.matchedLeaves=sampleLaunch(launchMatchedLeaves,samples);
    result.tensorLeaves=sampleLaunch(launchTensorLeaves,samples);

    launchCuda32();launchMatched();launchTensor();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Hit> hostCuda32(rayCount),hostMatched(rayCount);
    std::vector<Hit> hostTensor(rayCount),hostPhased(rayCount);
    CUDA_CHECK(cudaMemcpy(
        hostCuda32.data(),cuda32,rayCount*sizeof(Hit),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hostMatched.data(),matched,rayCount*sizeof(Hit),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hostTensor.data(),tensor,rayCount*sizeof(Hit),cudaMemcpyDeviceToHost));
    result.hitCount=int(std::count_if(
        hostMatched.begin(),hostMatched.end(),
        [](const Hit& hit){return hit.triangle>=0;}));
    result.cuda32Accuracy=
        compareHits(hostMatched,hostCuda32,scene.triangleCount);
    result.tensorAccuracy=
        compareHits(hostMatched,hostTensor,scene.triangleCount);
    result.cuda32Hits=hostCuda32;
    result.matchedHits=hostMatched;
    result.tensorHits=hostTensor;
    launchTensorLeaves();CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        hostPhased.data(),phased,rayCount*sizeof(Hit),cudaMemcpyDeviceToHost));
    result.phasedAccuracy=
        compareHits(hostMatched,hostPhased,scene.triangleCount);

    constexpr int strata=16;
    int bruteCount=std::min(256,rayCount);
    std::vector<Ray> hostBruteRays(bruteCount);
    std::vector<int> bruteIndices(bruteCount);
    if(rayCount==w*h&&bruteCount==strata*strata) {
        std::vector<int> rayPositionByPixel(w*h,-1);
        for(int position=0;position<rayCount;position++)
            rayPositionByPixel[originalPixels[position]]=position;
        int sample=0;
        for(int sy=0;sy<strata;sy++)for(int sx=0;sx<strata;sx++) {
            int x=(2*sx+1)*w/(2*strata);
            int y=(2*sy+1)*h/(2*strata);
            int position=rayPositionByPixel[y*w+x];
            bruteIndices[sample]=position;
            hostBruteRays[sample]=rays[position];
            sample++;
        }
    } else {
        std::vector<int> positions(rayCount);
        std::iota(positions.begin(),positions.end(),0);
        std::sort(
            positions.begin(),positions.end(),[&](int a,int b) {
                return originalPixels[a]<originalPixels[b];
            });
        for(int sample=0;sample<bruteCount;sample++) {
            int ordered=(2*sample+1)*rayCount/(2*bruteCount);
            int position=positions[ordered];
            bruteIndices[sample]=position;
            hostBruteRays[sample]=rays[position];
        }
    }
    CUDA_CHECK(cudaMemcpy(
        bruteRays,hostBruteRays.data(),bruteCount*sizeof(Ray),
        cudaMemcpyHostToDevice));
    bruteTrace<<<(bruteCount+255)/256,256>>>(
        brute,bruteRays,bruteCount,scene.triangles,scene.triangleCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Hit> hostBrute(bruteCount);
    CUDA_CHECK(cudaMemcpy(
        hostBrute.data(),brute,bruteCount*sizeof(Hit),cudaMemcpyDeviceToHost));
    std::vector<Hit> matchedSubset;
    matchedSubset.reserve(bruteCount);
    for(int index:bruteIndices)matchedSubset.push_back(hostMatched[index]);
    result.bruteAccuracy=
        compareHits(hostBrute,matchedSubset,scene.triangleCount);

    traceCuda32Wide<<<cudaBlocks,CUDA_BLOCK_THREADS>>>(
        cuda32,deviceRays,rayCount,scene.triangles,scene.nodes,cuda32Stats);
    traceMatchedWide<<<packets,32>>>(
        matched,deviceRays,rayCount,scene.triangles,scene.nodes,matchedStats);
    switch(tensorVariant) {
        case TensorVariant::Validated:
            traceTensorWide<TensorVariant::Validated><<<packets,32>>>(
                tensor,deviceRays,rayCount,scene.triangles,
                scene.coefficients,scene.frames,scene.nodes,tensorStats);
            break;
        case TensorVariant::UvtDepthSorted:
            traceTensorWide<
                TensorVariant::UvtDepthSorted><<<packets,32>>>(
                tensor,deviceRays,rayCount,scene.triangles,
                scene.coefficients,scene.frames,scene.nodes,tensorStats);
            break;
        case TensorVariant::E0E1E2:
            traceTensorWide<TensorVariant::E0E1E2><<<packets,32>>>(
                tensor,deviceRays,rayCount,scene.triangles,
                scene.coefficients,scene.frames,scene.nodes,tensorStats);
            break;
    }
    CUDA_CHECK(cudaMemset(overflow,0,sizeof(int)));
    collectPacketLeaves<<<packets,32>>>(
        deviceRays,rayCount,scene.nodes,leaves,leafCounts,overflow,
        traversalStats);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        &result.cuda32Stats,cuda32Stats,sizeof(DeviceStats),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &result.matchedStats,matchedStats,sizeof(DeviceStats),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &result.tensorStats,tensorStats,sizeof(DeviceStats),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &result.traversalStats,traversalStats,sizeof(DeviceStats),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &result.overflow,overflow,sizeof(int),cudaMemcpyDeviceToHost));
    std::vector<int> hostLeafCounts(packets);
    CUDA_CHECK(cudaMemcpy(
        hostLeafCounts.data(),leafCounts,packets*sizeof(int),
        cudaMemcpyDeviceToHost));
    result.maxLeaves=*std::max_element(
        hostLeafCounts.begin(),hostLeafCounts.end());

    cudaFree(deviceRays);cudaFree(cuda32);cudaFree(matched);
    cudaFree(tensor);cudaFree(phased);
    cudaFree(brute);cudaFree(bruteRays);cudaFree(leaves);
    cudaFree(leafCounts);cudaFree(overflow);
    cudaFree(cuda32Stats);cudaFree(matchedStats);
    cudaFree(tensorStats);cudaFree(traversalStats);
    return result;
}

static bool accuracyPass(
    const Accuracy& accuracy,bool requireSamePrimitive) {
    return accuracy.hitMiss==0&&accuracy.invalid==0&&
           accuracy.maxRelT<1e-4f&&
           (!requireSamePrimitive||accuracy.primitive==0);
}

static void printAccuracy(const char* label,const Accuracy& accuracy) {
    std::printf(
        "%s FP/FN=%d/%d primitive=%d invalid=%d max|dt|=%.3g rel=%.3g",
        label,accuracy.falsePositive,accuracy.falseNegative,
        accuracy.primitive,accuracy.invalid,accuracy.maxAbsT,
        accuracy.maxRelT);
}

static void writeRawSamples(
    FILE* output,const char* scene,const char* rayKind,const char* rayOrder,
    const char* bvhBuilder,const char* tensorVariant,int w,int h,
    int leafTriangles,const char* scope,const Timing& timing) {
    if(!output)return;
    for(size_t i=0;i<timing.samples.size();i++)
        std::fprintf(
            output,"%s,%s,%s,%s,%s,%d,%d,%d,%s,%zu,%.9g\n",
            scene,rayKind,rayOrder,bvhBuilder,tensorVariant,w,h,
            leafTriangles,scope,i,timing.samples[i]);
}

static std::string fileSlug(std::string value) {
    for(char& c:value)
        if(!std::isalnum(static_cast<unsigned char>(c)))c='-';
    return value;
}

static void writeHitImage(
    const std::string& path,const std::vector<Hit>& hits,
    const std::vector<int>& originalPixels,const DeviceScene& scene,
    const char* sceneName,int w,int h) {
    std::vector<Triangle> triangles(scene.triangleCount);
    CUDA_CHECK(cudaMemcpy(
        triangles.data(),scene.triangles,
        triangles.size()*sizeof(Triangle),cudaMemcpyDeviceToHost));
    std::vector<unsigned char> pixels(size_t(w)*h*3,18);
    for(size_t i=0;i<hits.size();i++) {
        if(hits[i].triangle<0||hits[i].triangle>=scene.triangleCount)continue;
        int pixel=originalPixels[i];
        int source=triangles[hits[i].triangle].sourcePrimitive;
        unsigned char r,g,b;
        if(!std::strcmp(sceneName,"Grid")) {
            int cell=source/2;
            int x=cell%128,z=cell/128;
            bool light=((x/8)+(z/8))%2==0;
            r=light?224:42;
            g=light?218:72;
            b=light?196:116;
        } else {
            uint32_t hash=uint32_t(source)*0x9e3779b9u;
            r=64+((hash>>0)&127);
            g=64+((hash>>8)&127);
            b=64+((hash>>16)&127);
        }
        pixels[size_t(pixel)*3+0]=r;
        pixels[size_t(pixel)*3+1]=g;
        pixels[size_t(pixel)*3+2]=b;
    }
    std::ofstream output(path,std::ios::binary);
    output<<"P6\n"<<w<<" "<<h<<"\n255\n";
    output.write(
        reinterpret_cast<const char*>(pixels.data()),
        std::streamsize(pixels.size()));
    if(!output) {
        std::fprintf(stderr,"Failed to write %s\n",path.c_str());
        return;
    }
    std::printf("    wrote %s\n",path.c_str());
}

static bool runScene(
    HostScene host,int w,int h,int samples,int leafTriangles,RayMode rayMode,
    BvhBuilder builderChoice,TensorVariant tensorVariant,
    uint64_t* expectedRayHash,FILE* rawCsv,
    const std::string& renderPrefix) {
    std::vector<WideNode> wide;
    double buildMs=0;
    BvhBuildReport buildReport{};
    DeviceScene scene=uploadScene(
        std::move(host.triangles),&wide,&buildMs,leafTriangles,builderChoice,
        tensorVariant,&buildReport);
    std::printf(
        "\n%s: %d source / %d packed triangles | %s -> BVH8 %d nodes | "
        "%d leaves, logical leaf min/mean/max %d/%.1f/%d | "
        "build+pack+upload %.1f ms | %s coefficient batches=%d\n",
        host.name.c_str(),scene.sourceTriangleCount,scene.triangleCount,
        buildReport.name.c_str(),scene.nodeCount,buildReport.leafCount,
        buildReport.minLeafTriangles,buildReport.meanLeafTriangles,
        buildReport.maxLeafTriangles,buildMs,
        tensorVariant==TensorVariant::Validated?
            "exact-only":"unrepresentable/rejected",
        scene.exactOnlyBatches);

    std::vector<int> basePixels;
    std::vector<Ray> baseRays=
        makePrimaryRays(w,h,host.camera,&basePixels);
    double secondarySetupMs=0;
    if(rayMode==RayMode::Secondary)
        baseRays=makeSecondaryRays(
            scene,baseRays,basePixels,&basePixels,&secondarySetupMs);
    if(baseRays.empty()) {
        std::fprintf(
            stderr,"%s produced no valid %s rays\n",
            host.name.c_str(),rayModeName(rayMode));
        freeScene(&scene);
        return false;
    }
    uint64_t workloadHash=raySetHash(baseRays,basePixels);
    if(*expectedRayHash&&*expectedRayHash!=workloadHash) {
        std::fprintf(
            stderr,
            "%s %s ray hash changed across BVH controls: %016llx != "
            "%016llx\n",
            host.name.c_str(),rayModeName(rayMode),
            static_cast<unsigned long long>(workloadHash),
            static_cast<unsigned long long>(*expectedRayHash));
        freeScene(&scene);
        return false;
    }
    *expectedRayHash=workloadHash;
    std::printf(
        "  rays: %s, %zu active of %d camera samples | set hash %016llx",
        rayModeName(rayMode),baseRays.size(),w*h,
        static_cast<unsigned long long>(workloadHash));
    if(rayMode==RayMode::Secondary)
        std::printf(
            " (primary trace + deterministic cosine bounce setup %.2f ms, "
            "excluded from trace timing)",
            secondarySetupMs);
    std::printf("\n");

    bool passed=true;
    for(bool packetShuffled:{false,true}) {
        std::vector<int> originalPixels=basePixels;
        std::vector<Ray> rays=baseRays;
        if(packetShuffled)shuffleRays(&rays,&originalPixels);
        BenchResult result=benchmarkMode(
            scene,rays,originalPixels,w,h,samples,tensorVariant);
        const char* rayOrder=
            packetShuffled?"packet-shuffled":
            rayMode==RayMode::Primary?"coherent":"pixel-ordered";
        const char* variantName=tensorVariantName(tensorVariant);
        if(!packetShuffled&&!renderPrefix.empty()) {
            std::string base=renderPrefix+"-"+
                fileSlug(buildReport.name);
            writeHitImage(
                base+"-cuda32.ppm",result.cuda32Hits,originalPixels,scene,
                host.name.c_str(),w,h);
            writeHitImage(
                base+"-cuda-packet16.ppm",result.matchedHits,originalPixels,scene,
                host.name.c_str(),w,h);
            std::string tensorSuffix=
                tensorVariant==TensorVariant::Validated?
                "-wmma-validated.ppm":"-"+std::string(variantName)+".ppm";
            writeHitImage(
                base+tensorSuffix,result.tensorHits,originalPixels,scene,
                host.name.c_str(),w,h);
        }
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "integrated-cuda32",result.cuda32);
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "integrated-matched",result.matched);
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "integrated-tensor",result.tensor);
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "separated-traversal",result.traversal);
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "separated-matched-leaves",result.matchedLeaves);
        writeRawSamples(
            rawCsv,host.name.c_str(),rayModeName(rayMode),rayOrder,
            buildReport.name.c_str(),variantName,w,h,leafTriangles,
            "separated-tensor-leaves",result.tensorLeaves);
        float integratedSpeedup=result.cuda32.median/result.tensor.median;
        float matched16Speedup=result.matched.median/result.tensor.median;
        float leafSpeedup=
            result.matchedLeaves.median/result.tensorLeaves.median;
        float matchedPhased=result.traversal.median+result.matchedLeaves.median;
        float tensorPhased=result.traversal.median+result.tensorLeaves.median;
        std::printf("  %-15s %dx%d (%zu rays)\n",
                    rayOrder,w,h,rays.size());
        std::printf(
            "    integrated: CUDA32 %.4f [%.4f,%.4f] ms | "
            "CUDA-packet16 %.4f [%.4f,%.4f] ms | "
            "WMMA(F16->F32)/%s %.4f [%.4f,%.4f] ms\n",
            result.cuda32.median,result.cuda32.p10,result.cuda32.p90,
            result.matched.median,result.matched.p10,result.matched.p90,
            variantName,
            result.tensor.median,result.tensor.p10,result.tensor.p90);
        std::printf(
            "    speedup: WMMA vs CUDA32 %.3fx | "
            "vs diagnostic CUDA-packet16 %.3fx\n",
            integratedSpeedup,matched16Speedup);
        std::printf(
            "    separated fixed-work: traversal %.4f ms | CUDA leaves "
            "%.4f ms | WMMA leaves %.4f ms (%.3fx) | diagnostic sum "
            "%.4f/%.4f ms\n",
            result.traversal.median,result.matchedLeaves.median,
            result.tensorLeaves.median,leafSpeedup,matchedPhased,tensorPhased);
        double raysCount=rays.size();
        if(tensorVariant==TensorVariant::Validated)
            std::printf(
                "    CUDA32 work: %.2f nodes/ray %.2f leaves/ray "
                "%.1f triangle tests/ray | WMMA survivors to Moller %.1f/ray | "
                "numeric fallbacks=%llu | hits=%d | packet leaves max=%d "
                "overflow=%d | stack overflow CUDA32/packet16/WMMA/phase="
                "%llu/%llu/%llu/%llu\n",
                double(result.cuda32Stats.nodeVisits)/raysCount,
                double(result.cuda32Stats.rayLeafVisits)/raysCount,
                double(result.cuda32Stats.triangleTests)/raysCount,
                double(result.tensorStats.exactTests)/raysCount,
                result.tensorStats.numericFallbacks,result.hitCount,
                result.maxLeaves,result.overflow,
                result.cuda32Stats.stackOverflows,
                result.matchedStats.stackOverflows,
                result.tensorStats.stackOverflows,
                result.traversalStats.stackOverflows);
        else
            std::printf(
                "    CUDA32 work: %.2f nodes/ray %.2f leaves/ray "
                "%.1f triangle tests/ray\n"
                "    WMMA work: %.2f leaves/ray %.1f triangle pairs/ray | "
                "depth updates %.2f/ray | "
                "Moller checks=0 | reference hits=%d | packet leaves max=%d "
                "overflow=%d | stack overflow CUDA32/packet16/WMMA/phase="
                "%llu/%llu/%llu/%llu\n",
                double(result.cuda32Stats.nodeVisits)/raysCount,
                double(result.cuda32Stats.rayLeafVisits)/raysCount,
                double(result.cuda32Stats.triangleTests)/raysCount,
                double(result.tensorStats.rayLeafVisits)/raysCount,
                double(result.tensorStats.triangleTests)/raysCount,
                double(result.tensorStats.exactTests)/raysCount,
                result.hitCount,result.maxLeaves,result.overflow,
                result.cuda32Stats.stackOverflows,
                result.matchedStats.stackOverflows,
                result.tensorStats.stackOverflows,
                result.traversalStats.stackOverflows);
        std::printf("    correctness: ");
        printAccuracy("CUDA32",result.cuda32Accuracy);
        std::printf(" | ");
        printAccuracy("WMMA",result.tensorAccuracy);
        std::printf(" | ");
        printAccuracy("phased",result.phasedAccuracy);
        std::printf(" | ");
        printAccuracy("BVH/brute-256",result.bruteAccuracy);
        if(tensorVariantIsApproximate(tensorVariant))
            std::printf(
                " | approximate rates FP/rays=%.4f%% FN/reference-hits="
                "%.4f%% wrong-primitive/reference-hits=%.4f%%",
                100.*result.tensorAccuracy.falsePositive/raysCount,
                100.*result.tensorAccuracy.falseNegative/
                    std::max(1,result.hitCount),
                100.*result.tensorAccuracy.primitive/
                    std::max(1,result.hitCount));
        // Approximate variants report image disagreement without treating it
        // as a harness failure. Baseline correctness, finite/index-safe Tensor
        // output, and traversal integrity remain mandatory.
        bool modePass=accuracyPass(result.cuda32Accuracy,true)&&
                      accuracyPass(result.bruteAccuracy,true)&&
                      !result.overflow&&
                      !result.cuda32Stats.stackOverflows&&
                      !result.matchedStats.stackOverflows&&
                      !result.tensorStats.stackOverflows&&
                      !result.traversalStats.stackOverflows;
        if(tensorVariant==TensorVariant::Validated)
            modePass&=accuracyPass(result.tensorAccuracy,true)&&
                      accuracyPass(result.phasedAccuracy,true);
        else
            modePass&=!result.tensorAccuracy.invalid&&
                      !result.phasedAccuracy.invalid&&
                      (!result.hitCount||
                       (result.tensorAccuracy.falseNegative<result.hitCount&&
                        result.phasedAccuracy.falseNegative<result.hitCount));
        if(host.name=="RegressionTinyFar"&&
           tensorVariant==TensorVariant::Validated)
            modePass&=result.hitCount>0&&
                      result.tensorStats.numericFallbacks>=4;
        if(host.name=="RegressionDepthSorted")
            modePass&=result.hitCount>0&&
                      result.tensorAccuracy.hitMiss==0&&
                      result.tensorAccuracy.primitive==0&&
                      result.tensorAccuracy.invalid==0&&
                      result.tensorAccuracy.maxRelT<.02f&&
                      result.phasedAccuracy.hitMiss==0&&
                      result.phasedAccuracy.primitive==0&&
                      result.phasedAccuracy.invalid==0&&
                      result.phasedAccuracy.maxRelT<.02f;
        if(host.name=="RegressionDepthClip")
            modePass&=result.hitCount>0&&
                      result.tensorAccuracy.hitMiss==0&&
                      result.tensorAccuracy.primitive==0&&
                      result.tensorAccuracy.invalid==0&&
                      result.tensorAccuracy.maxRelT<.02f&&
                      result.phasedAccuracy.hitMiss==0&&
                      result.phasedAccuracy.primitive==0&&
                      result.phasedAccuracy.invalid==0&&
                      result.phasedAccuracy.maxRelT<.02f;
        // This fixture has a guaranteed near/far two-leaf topology only for
        // the built-in builder at leaf size four. Other configurations still
        // check Tensor depth, but cannot assert a particular pruning count.
        if(host.name=="RegressionDepthClip"&&leafTriangles==4&&
           builderChoice==BvhBuilder::BuiltinSah)
            modePass&=result.tensorStats.rayLeafVisits<
                          result.traversalStats.rayLeafVisits&&
                      result.matchedStats.rayLeafVisits<
                          result.traversalStats.rayLeafVisits;
        passed&=modePass;
        std::printf(
            " => %s%s\n",modePass?"PASS":"FAIL",
            tensorVariantIsApproximate(tensorVariant)?
            " (approximate accuracy is informational)":"");
    }
    freeScene(&scene);
    return passed;
}

static HostScene makeGridScene() {
    HostScene scene{};
    scene.name="Grid";
    constexpr int side=128;
    scene.triangles.reserve(side*side*2);
    for(int z=0;z<side;z++)for(int x=0;x<side;x++) {
        auto point=[&](int px,int pz) {
            float fx=2.f*px/side-1.f,fz=2.f*pz/side-1.f;
            float y=.08f*sinf(fx*17.f)*cosf(fz*13.f);
            return Vec3{fx,y,fz};
        };
        Vec3 a=point(x,z),b=point(x+1,z),c=point(x+1,z+1),d=point(x,z+1);
        scene.triangles.push_back({a,sub(b,a),sub(c,a)});
        scene.triangles.push_back({a,sub(c,a),sub(d,a)});
    }
    scene.camera={{1.7f,1.4f,1.7f},{0,0,0},{0,1,0}};
    return scene;
}

static HostScene makeOddLeafRegressionScene(int triangleCount) {
    HostScene scene{};
    scene.name="RegressionOdd"+std::to_string(triangleCount);
    scene.triangles.reserve(triangleCount);
    for(int i=0;i<triangleCount;i++) {
        float x=-.75f+1.5f*(i+.5f)/triangleCount;
        Vec3 a{x-.08f,-.18f,0};
        Vec3 c{x+.08f,-.18f,0};
        Vec3 b{x,.18f,0};
        scene.triangles.push_back({a,sub(c,a),sub(b,a)});
    }
    scene.camera={{0,0,2},{0,0,0},{0,1,0}};
    return scene;
}

static HostScene makeTinyFarRegressionScene() {
    HostScene scene{};
    scene.name="RegressionTinyFar";
    auto addTriangle=[&](Vec3 a,Vec3 c,Vec3 b) {
        scene.triangles.push_back({a,sub(c,a),sub(b,a)});
    };
    addTriangle(
        {-5e-4f,-5e-4f,0},{5e-4f,-5e-4f,0},{0,5e-4f,0});
    addTriangle(
        {6e-4f,-6e-4f,0},{8e-4f,-6e-4f,0},{7e-4f,-4e-4f,0});
    addTriangle(
        {-8e-4f,4e-4f,0},{-6e-4f,4e-4f,0},{-7e-4f,6e-4f,0});
    addTriangle(
        {6e-4f,4e-4f,0},{8e-4f,4e-4f,0},{7e-4f,6e-4f,0});
    scene.camera={{0,0,100},{0,0,0},{0,1,0}};
    return scene;
}

static HostScene makeDepthSortedRegressionScene() {
    HostScene scene{};
    scene.name="RegressionDepthSorted";
    auto addTriangle=[&](float z) {
        Vec3 a{-.8f,-.8f,z},c{.8f,-.8f,z},b{0,.8f,z};
        scene.triangles.push_back({a,sub(c,a),sub(b,a)});
    };
    // Deliberately store the farther triangle first. Tensor depth must replace
    // it with the second triangle in the same leaf.
    addTriangle(0.f);
    addTriangle(.25f);
    scene.camera={{0,0,2},{0,0,0},{0,1,0}};
    return scene;
}

static HostScene makeDepthClipRegressionScene() {
    HostScene scene{};
    scene.name="RegressionDepthClip";
    auto addLayer=[&](float z) {
        for(int i=0;i<4;i++) {
            Vec3 a{-.8f,-.8f,z},c{.8f,-.8f,z},b{0,.8f,z};
            scene.triangles.push_back({a,sub(c,a),sub(b,a)});
        }
    };
    // Looking in +Z makes the SAH leaf order near-to-far. The near layer must
    // set world-space hit.t so the later far leaf is rejected by its AABB.
    addLayer(-.25f);
    addLayer(.25f);
    scene.camera={{0,0,-2},{0,0,0},{0,1,0}};
    return scene;
}

static bool inputAvailable(const char* path) {
    if(!path||!*path)return false;
    std::ifstream input(path,std::ios::binary);
    return input.good();
}

static void printUsage(const char* executable) {
    std::printf(
        "RayMMA %s\n"
        "usage: %s [options]\n"
        "  --quick                 128x72, five timing samples\n"
        "  --resolution WxH        set ray image dimensions\n"
        "  --leaf N                maximum triangles/leaf (multiple of four)\n"
        "  --candidate-rich        alias for --leaf 256\n"
        "  --leaf-sweep            run 4,8,16,32,64,128,256\n"
        "  --ray-mode MODE         primary or secondary\n"
        "  --variant NAME          validated (default mixed-WMMA filter +\n"
        "                          FP32 Moller-Trumbore), uvt-depthsorted, or\n"
        "                          e0e1e2\n"
        "                          (the last two are no-Moller, Tensor-owned,\n"
        "                          and intentionally approximate)\n"
        "  --bvh BUILDER           builtin, tinybvh-sah, or tinybvh-sbvh\n"
        "  --bvh-sweep             run builtin and both TinyBVH builders\n"
        "  --render-prefix PATH    write CUDA32/CUDA-packet16/WMMA PPM images\n"
        "  --raw-csv FILE          write every CUDA-event sample\n"
        "  --scene NAME            Grid, CoastalCliffLow/Mid/High, Sibenik,\n"
        "                          Sponza, or SanMiguel\n"
        "                          (RegressionOdd5/6/7 and\n"
        "                          RegressionTinyFar/DepthSorted/DepthClip\n"
        "                          are fixtures)\n"
        "  --include-sanmiguel     add SanMiguel to the default suite\n"
        "  --help                  show this help\n"
        "  --version               show the version\n",
        RAYMMA_VERSION,executable);
}

int main(int argc,char** argv) {
    int w=256,h=144,samples=9;
    int leafTriangles=DEFAULT_LEAF_TRIANGLES;
    bool includeSanMiguel=false;
    bool leafSweep=false;
    bool bvhSweep=false;
    RayMode rayMode=RayMode::Primary;
    TensorVariant tensorVariant=TensorVariant::Validated;
    BvhBuilder bvhBuilder=BvhBuilder::BuiltinSah;
    std::string only,rawCsvPath,renderPrefix;
    for(int i=1;i<argc;i++) {
        if(!std::strcmp(argv[i],"--help")||!std::strcmp(argv[i],"-h")) {
            printUsage(argv[0]);
            return 0;
        }
        else if(!std::strcmp(argv[i],"--version")) {
            std::printf("RayMMA %s\n",RAYMMA_VERSION);
            return 0;
        }
        else if(!std::strcmp(argv[i],"--include-sanmiguel"))
            includeSanMiguel=true;
        else if(!std::strcmp(argv[i],"--candidate-rich"))leafTriangles=256;
        else if(!std::strcmp(argv[i],"--leaf-sweep"))leafSweep=true;
        else if(!std::strcmp(argv[i],"--bvh-sweep"))bvhSweep=true;
        else if(!std::strcmp(argv[i],"--ray-mode")&&i+1<argc) {
            const char* value=argv[++i];
            if(!std::strcmp(value,"primary"))rayMode=RayMode::Primary;
            else if(!std::strcmp(value,"secondary"))
                rayMode=RayMode::Secondary;
            else return std::fprintf(
                stderr,"Ray mode must be primary or secondary\n"),2;
        }
        else if(!std::strcmp(argv[i],"--variant")&&i+1<argc) {
            const char* value=argv[++i];
            if(!std::strcmp(value,"validated"))
                tensorVariant=TensorVariant::Validated;
            else if(!std::strcmp(value,"uvt-depthsorted"))
                tensorVariant=TensorVariant::UvtDepthSorted;
            else if(!std::strcmp(value,"e0e1e2"))
                tensorVariant=TensorVariant::E0E1E2;
            else return std::fprintf(
                stderr,"Unknown WMMA variant: %s\n",value),2;
        }
        else if(!std::strcmp(argv[i],"--bvh")&&i+1<argc) {
            const char* value=argv[++i];
            if(!std::strcmp(value,"builtin"))
                bvhBuilder=BvhBuilder::BuiltinSah;
            else if(!std::strcmp(value,"tinybvh-sah"))
                bvhBuilder=BvhBuilder::TinyBvhSah;
            else if(!std::strcmp(value,"tinybvh-sbvh"))
                bvhBuilder=BvhBuilder::TinyBvhSbvh;
            else return std::fprintf(
                stderr,"Unknown BVH builder: %s\n",value),2;
        }
        else if(!std::strcmp(argv[i],"--raw-csv")&&i+1<argc)
            rawCsvPath=argv[++i];
        else if(!std::strcmp(argv[i],"--render-prefix")&&i+1<argc)
            renderPrefix=argv[++i];
        else if(!std::strcmp(argv[i],"--quick")){w=128;h=72;samples=5;}
        else if(!std::strcmp(argv[i],"--resolution")&&i+1<argc) {
            if(std::sscanf(argv[++i],"%dx%d",&w,&h)!=2||w<16||h<16)
                return std::fprintf(stderr,"Invalid resolution\n"),2;
        }
        else if(!std::strcmp(argv[i],"--leaf")&&i+1<argc) {
            leafTriangles=std::atoi(argv[++i]);
            if(leafTriangles<4||leafTriangles%4)
                return std::fprintf(
                    stderr,"Leaf size must be a positive multiple of 4\n"),2;
        }
        else if(!std::strcmp(argv[i],"--scene")&&i+1<argc)only=argv[++i];
        else {
            printUsage(argv[0]);
            return 2;
        }
    }
    int deviceCount=0;CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(!deviceCount)return std::fprintf(stderr,"No CUDA device\n"),1;
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
    if(properties.major<7)
        return std::fprintf(
            stderr,
            "RayMMA requires compute capability 7.0 or newer; found %d.%d\n",
            properties.major,properties.minor),1;
    int driverVersion=0,runtimeVersion=0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    std::printf(
        "RayMMA %s (%s) on %s, compute %d.%d\n"
        "CUDA driver API %d.%d, runtime %d.%d\n"
        "%s -> BVH8, <=%d-triangle leaves, %s rays, WMMA variant=%s, "
        "%d samples\n",
        RAYMMA_VERSION,RAYMMA_BUILD_CONFIG,
        properties.name,properties.major,properties.minor,
        driverVersion/1000,(driverVersion%1000)/10,
        runtimeVersion/1000,(runtimeVersion%1000)/10,
        bvhSweep?"BVH builder sweep":bvhBuilderName(bvhBuilder),
        leafTriangles,rayModeName(rayMode),tensorVariantName(tensorVariant),
        samples);
    if((bvhSweep||bvhBuilder!=BvhBuilder::BuiltinSah)&&
       !tinyBvhAvailable())
        return std::fprintf(
            stderr,
            "TinyBVH was requested, but this binary was configured without "
            "RAYMMA_ENABLE_TINYBVH=ON\n"),2;

    std::vector<std::function<HostScene()>> loaders;
    auto add=[&](const char* name,std::function<HostScene()> loader) {
        if(only.empty()||only==name)loaders.push_back(std::move(loader));
    };
    add("Grid",[]{return makeGridScene();});
    if(only=="RegressionOdd5")
        loaders.push_back([]{return makeOddLeafRegressionScene(5);});
    if(only=="RegressionOdd6")
        loaders.push_back([]{return makeOddLeafRegressionScene(6);});
    if(only=="RegressionOdd7")
        loaders.push_back([]{return makeOddLeafRegressionScene(7);});
    if(only=="RegressionTinyFar")
        loaders.push_back([]{return makeTinyFarRegressionScene();});
    if(only=="RegressionDepthSorted")
        loaders.push_back([]{return makeDepthSortedRegressionScene();});
    if(only=="RegressionDepthClip")
        loaders.push_back([]{return makeDepthClipRegressionScene();});
    if(inputAvailable(COASTAL_CLIFF_LOW_PATH))
        add("CoastalCliffLow",[]{return loadBrtri(
            COASTAL_CLIFF_LOW_PATH,"CoastalCliffLow",
            {{.8f,1.15f,2.05f},{0,0,0},{0,1,0}});});
    if(inputAvailable(COASTAL_CLIFF_MID_PATH))
        add("CoastalCliffMid",[]{return loadBrtri(
            COASTAL_CLIFF_MID_PATH,"CoastalCliffMid",
            {{.8f,1.15f,2.05f},{0,0,0},{0,1,0}});});
    if(inputAvailable(COASTAL_CLIFF_HIGH_PATH))
        add("CoastalCliffHigh",[]{return loadBrtri(
            COASTAL_CLIFF_HIGH_PATH,"CoastalCliffHigh",
            {{.8f,1.15f,2.05f},{0,0,0},{0,1,0}});});
    if(inputAvailable(SIBENIK_PATH))
        add("Sibenik",[]{return loadObj(
            SIBENIK_PATH,"Sibenik",
            {{-15.5f,-2.5f,0},{-14.5f,-2.5f,0},{0,1,0}});});
    if(inputAvailable(SPONZA_PATH))
        add("Sponza",[]{return loadObj(
            SPONZA_PATH,"Sponza",
            {{800,580,-35},{799,580,-35},{0,1,0}});});
    if(includeSanMiguel&&inputAvailable(SANMIGUEL_PATH))
        add("SanMiguel",[]{return loadObj(
            SANMIGUEL_PATH,"SanMiguel",
            {{8,1.5f,10.5f},{9,1.5f,9.5f},{0,1,0}});});
    if(loaders.empty())
        return std::fprintf(stderr,"Unknown or unavailable scene: %s\n",
                            only.c_str()),2;
    FILE* rawCsv=nullptr;
    if(!rawCsvPath.empty()) {
        rawCsv=std::fopen(rawCsvPath.c_str(),"w");
        if(!rawCsv)
            return std::fprintf(
                stderr,"Cannot open raw CSV: %s\n",rawCsvPath.c_str()),2;
        std::fprintf(
            rawCsv,
            "scene,ray_kind,ray_order,bvh_builder,tensor_variant,width,height,"
            "max_leaf_triangles,"
            "timing_scope,sample_index,milliseconds\n");
    }
    bool passed=true;
    std::vector<BvhBuilder> bvhBuilders;
    if(bvhSweep)
        bvhBuilders={
            BvhBuilder::BuiltinSah,
            BvhBuilder::TinyBvhSah,
            BvhBuilder::TinyBvhSbvh};
    else bvhBuilders={bvhBuilder};
    for(auto& loader:loaders) {
        uint64_t expectedRayHash=0;
        for(BvhBuilder builder:bvhBuilders)
            if(leafSweep) {
                for(int leaf:{4,8,16,32,64,128,256})
                    passed&=runScene(
                        loader(),w,h,samples,leaf,rayMode,builder,tensorVariant,
                        &expectedRayHash,rawCsv,renderPrefix);
            } else {
                passed&=runScene(
                    loader(),w,h,samples,leafTriangles,rayMode,builder,
                    tensorVariant,
                    &expectedRayHash,rawCsv,renderPrefix);
            }
    }
    if(rawCsv)std::fclose(rawCsv);
    if(tensorVariantIsApproximate(tensorVariant))
        std::printf(
            "\nresearch harness => %s (approximate image accuracy is "
            "reported above)\n",passed?"PASS":"FAIL");
    else std::printf("\nresearch suite => %s\n",passed?"PASS":"FAIL");
    return passed?0:1;
}

// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

constexpr int BVH_WIDTH=8;

struct Vec3 { float x,y,z; };
struct Triangle {
    Vec3 a,c,b;
    int sourcePrimitive=-1;
};

struct WideNode {
    Vec3 lo[BVH_WIDTH];
    Vec3 hi[BVH_WIDTH];
    int child[BVH_WIDTH];
    int first[BVH_WIDTH];
    int count[BVH_WIDTH];
    int childCount=0;
};

enum class BvhBuilder {
    BuiltinSah,
    TinyBvhSah,
    TinyBvhSbvh
};

struct BvhBuildReport {
    std::string name;
    int sourceTriangles=0;
    int packedTriangles=0;
    int leafCount=0;
    int minLeafTriangles=0;
    int maxLeafTriangles=0;
    double meanLeafTriangles=0;
    float sahCost=0;
};

const char* bvhBuilderName(BvhBuilder builder);
bool tinyBvhAvailable();
bool buildTinyBvh(
    BvhBuilder builder,int maxLeafTriangles,
    std::vector<Triangle>* triangles,std::vector<WideNode>* nodes,
    BvhBuildReport* report,std::string* error);

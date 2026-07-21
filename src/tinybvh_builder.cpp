// SPDX-License-Identifier: MIT
//
// TinyBVH is an optional, separately obtained MIT dependency. This bridge
// converts its binned-SAH or spatial-split hierarchy into RayMMA's common
// BVH8 layout. Both CUDA controls and the Tensor path then traverse exactly
// the same nodes and reordered triangles.
#include "production_bvh.h"

#define NO_THREADED_BUILDS
#define TINYBVH_NO_SIMD
#define TINYBVH_IMPLEMENTATION
#include <tiny_bvh.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

using TinyWide=tinybvh::MBVH<8>;

Vec3 fromTiny(tinybvh::bvhvec3 value) {
    return {value.x,value.y,value.z};
}

int emitNode(
    uint32_t inputIndex,const TinyWide& input,
    const std::vector<Triangle>& source,std::vector<Triangle>* packed,
    std::vector<WideNode>* output,BvhBuildReport* report) {
    int outputIndex=int(output->size());
    output->emplace_back();
    const TinyWide::MBVHNode& inputNode=input.mbvhNode[inputIndex];
    WideNode node{};
    node.childCount=int(inputNode.childCount);
    for(int slot=0;slot<node.childCount;slot++) {
        uint32_t childIndex=inputNode.child[slot];
        const TinyWide::MBVHNode& child=input.mbvhNode[childIndex];
        node.lo[slot]=fromTiny(child.aabbMin);
        node.hi[slot]=fromTiny(child.aabbMax);
        if(child.isLeaf()) {
            int first=int(packed->size());
            for(uint32_t i=0;i<child.triCount;i++) {
                uint32_t primitive=
                    input.bvh.primIdx[child.firstTri+i];
                if(primitive>=source.size())return -1;
                packed->push_back(source[primitive]);
            }
            int logicalCount=int(child.triCount);
            while(packed->size()%4)
                packed->push_back(
                    {{0,0,0},{0,0,0},{0,0,0},-1});
            node.child[slot]=-1;
            node.first[slot]=first;
            node.count[slot]=int(packed->size())-first;
            report->leafCount++;
            report->minLeafTriangles=
                std::min(report->minLeafTriangles,logicalCount);
            report->maxLeafTriangles=
                std::max(report->maxLeafTriangles,logicalCount);
            report->meanLeafTriangles+=logicalCount;
        } else {
            int converted=emitNode(
                childIndex,input,source,packed,output,report);
            if(converted<0)return -1;
            node.child[slot]=converted;
        }
    }
    (*output)[outputIndex]=node;
    return outputIndex;
}

} // namespace

const char* bvhBuilderName(BvhBuilder builder) {
    switch(builder) {
        case BvhBuilder::BuiltinSah:return "builtin-binned-SAH";
        case BvhBuilder::TinyBvhSah:return "TinyBVH-binned-SAH";
        case BvhBuilder::TinyBvhSbvh:return "TinyBVH-spatial-split";
    }
    return "unknown";
}

bool tinyBvhAvailable() { return true; }

bool buildTinyBvh(
    BvhBuilder builder,int maxLeafTriangles,
    std::vector<Triangle>* triangles,std::vector<WideNode>* nodes,
    BvhBuildReport* report,std::string* error) {
    if(builder==BvhBuilder::BuiltinSah) {
        *error="TinyBVH bridge received the builtin builder";
        return false;
    }
    if(triangles->empty()) {
        *error="TinyBVH cannot build an empty scene";
        return false;
    }

    std::vector<Triangle> source=*triangles;
    std::vector<tinybvh::bvhvec4> vertices;
    vertices.reserve(source.size()*3);
    for(const Triangle& triangle:source) {
        Vec3 a=triangle.a;
        Vec3 c{
            triangle.a.x+triangle.c.x,
            triangle.a.y+triangle.c.y,
            triangle.a.z+triangle.c.z};
        Vec3 b{
            triangle.a.x+triangle.b.x,
            triangle.a.y+triangle.b.y,
            triangle.a.z+triangle.b.z};
        vertices.emplace_back(a.x,a.y,a.z,0);
        vertices.emplace_back(c.x,c.y,c.z,0);
        vertices.emplace_back(b.x,b.y,b.z,0);
    }

    TinyWide hierarchy;
    hierarchy.settings.useSpatialSplits=
        builder==BvhBuilder::TinyBvhSbvh;
    hierarchy.settings.postOptimize=true;
    hierarchy.settings.optimizeIterations=25;
    hierarchy.Build(vertices.data(),uint32_t(source.size()));
    hierarchy.bvh.SplitLeafs(uint32_t(maxLeafTriangles));
    hierarchy.ConvertFrom(hierarchy.bvh,true);

    *report={};
    report->name=bvhBuilderName(builder);
    report->sourceTriangles=int(source.size());
    report->minLeafTriangles=std::numeric_limits<int>::max();
    report->sahCost=hierarchy.SAHCost();
    triangles->clear();
    nodes->clear();
    if(emitNode(0,hierarchy,source,triangles,nodes,report)<0) {
        *error="TinyBVH returned an invalid primitive index";
        return false;
    }
    report->packedTriangles=int(triangles->size());
    if(report->leafCount)
        report->meanLeafTriangles/=report->leafCount;
    else report->minLeafTriangles=0;
    return true;
}

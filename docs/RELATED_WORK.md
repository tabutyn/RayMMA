# Related work and novelty boundary

RayMMA does not claim that Tensor-Core ray/triangle intersection is novel.
The public search and implementation review performed for this
artifact were sufficient to find adjacent foundations, not to establish an
exhaustive novelty or patent opinion.

The method builds on well-established ideas:

- Möller and Trumbore's minimum-storage ray/triangle intersection provides the
  FP32 validation predicate:
  [DOI 10.1080/10867651.1997.10487468](https://doi.org/10.1080/10867651.1997.10487468).
- The separated feature vector contains the direction and moment of a ray,
  making it Plücker-style line algebra rather than a new representation of a
  ray.
- NVIDIA's WMMA interface maps a warp-cooperative `m16n16k16` matrix operation
  to Tensor Cores:
  [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/).
- Markidis et al. characterize Tensor Core programmability, throughput, and
  mixed-precision error:
  [arXiv:1803.04014](https://arxiv.org/abs/1803.04014).
- Woop, Benthin, and Wald show why ray/triangle boundary behavior needs more
  than ordinary floating-point predicates:
  [Watertight Ray/Triangle Intersection](https://jcgt.org/published/0002/01/05/).

The artifact documents a four-triangle by sixteen-ray coefficient layout, its
integration with candidate-rich BVH leaves, and its measured crossover with
FP32 revalidation. No novelty claim is made.

## Baselines not included

RayMMA compares CUDA software intersection strategies. It does not
benchmark NVIDIA RT Cores, OptiX, Vulkan ray tracing, DirectX Raytracing, or a
compressed production GPU traversal kernel. TinyBVH is used only as an
optional established hierarchy builder; its output is converted to RayMMA's
uncompressed BVH8 traversal format.

The repository does not include a raw all-pairs `N rays × M triangles` sweep.
Measurements cover Ampere and Hopper, but the single procedural cloud workload
does not support a broad architectural conclusion.

## Novelty and IP scope

The references above are not an exhaustive literature or patent search.
RayMMA makes no patent-novelty conclusion and is not legal advice.

# Related work and novelty boundary

RayMMA should not currently claim that Tensor-Core ray/triangle intersection
is novel. The public search and implementation review performed for this
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

The potentially distinctive artifact contribution is the particular
four-triangle by sixteen-ray coefficient layout, its integration with
candidate-rich BVH leaves, and the measured crossover with FP32 revalidation.
That is a research hypothesis, not a novelty conclusion.

## Baselines not included

RayMMA compares CUDA software intersection strategies. It does not currently
benchmark NVIDIA RT Cores, OptiX, Vulkan ray tracing, DirectX Raytracing, or a
compressed production GPU traversal kernel. TinyBVH is used only as an
optional established hierarchy builder; its output is converted to RayMMA's
uncompressed BVH8 traversal format.

The repository still lacks a raw all-pairs `N rays × M triangles` sweep. Its
RTX 3050 Ti plus paid Lambda A10, A100, and H100 measurements now span Ampere
and Hopper, but one small procedural workload is not enough for a broad
architectural conclusion. A 12-hour Lambda B200 availability watch found no
capacity, so Blackwell remains unmeasured.

Before a paper or patent claim:

1. conduct a structured literature and patent search for matrix-form,
   Plücker-form, packet, and Tensor-Core ray/primitive intersection;
2. compare equations and data layouts claim-by-claim, not only titles;
3. retain the new CUDA32, TinyBVH, and true-secondary controls across
   standard scenes and larger sample counts;
4. test additional GPU generations and workloads, including B200 when it is
   available; and
5. obtain qualified IP advice if patent protection matters before public
   disclosure.

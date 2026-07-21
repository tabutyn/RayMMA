# LinkedIn post

Can NVIDIA Tensor Cores accelerate ray–triangle intersection?

I built RayMMA to find out—and the most useful result was learning exactly
where they do not win.

RayMMA maps 4 triangles × 16 rays into an `m16n16k16` WMMA operation: 64
candidate intersections per Tensor Core tile, with FP16 inputs and FP32
accumulation.

To test it honestly, I also built:

- an independent-ray CUDA32 Möller–Trumbore baseline;
- a matched 16-ray CUDA packet diagnostic;
- shared SAH/BVH8 traversal and candidate ordering;
- Tensor-derived barycentrics, closest depth, and BVH clipping;
- coherent, shuffled, and secondary-ray workloads; and
- raw CUDA-event timing plus hit, primitive, and depth validation.

The result on an RTX 3050 Ti Laptop GPU:

- With selective 16-triangle BVH leaves, CUDA32 won every primary and
  secondary comparison on the new 461,824-triangle CC0 scene.
- With deliberately dense 256-triangle leaves, the no-Möller Tensor modes
  reached 1.07–1.29× the same-tree CUDA32 performance.
- That speed came with measurable approximation: 2–3 missed hits out of 970,
  3–5 wrong closest primitives, and up to 3.28% relative depth error.
- The exact validated Tensor/CUDA hybrid crossed CUDA32 in some coarse cases,
  but the best selective CUDA32 configuration was still about 7.8× faster for
  coherent rays and 3.3× faster for shuffled rays.

So the conclusion is not “Tensor Cores beat ray tracing.” It is narrower and,
I think, more useful:

Dense ray/triangle arithmetic can benefit from Tensor Cores, but candidate
generation and BVH selectivity dominate the complete system. A faster leaf
kernel is not automatically a faster tracer.

The project also forced me into the details I enjoy most: warp-level data
layout, mixed-precision numerical behavior, coordinate normalization,
closest-hit depth propagation, traversal correctness, benchmark design, and
GPU driver/device debugging.

All 15 tests pass. The repository includes the CUDA source, both approximate
variants, exact source and asset hashes, raw samples, complete transcripts,
REUSE 3.3 licensing, and the negative results—not just the favorable numbers.

This is the kind of GPU systems engineering work I want to keep doing.

Repository: https://github.com/tabutyn/RayMMA

I would be glad to compare results with anyone working on CUDA, ray tracing,
Tensor Core scheduling, or compacted ray work queues—especially on newer GPU
architectures.

#CUDA #GPUComputing #RayTracing #TensorCores #SystemsEngineering #NVIDIA

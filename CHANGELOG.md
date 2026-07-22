# Changelog

## 0.1.0 (2026-07-22)

- Implemented an FP16-input, FP32-accumulated WMMA layout evaluating four
  triangles against sixteen rays per tile.
- Added independent-ray CUDA32 and matched CUDA-packet16 controls, a validated
  WMMA filter with FP32 Möller–Trumbore ownership, and the approximate
  `uvt-depthsorted` and `e0e1e2` Tensor-owned variants.
- Integrated the backends with a shared uncompressed BVH8, maximum-leaf sweeps,
  coherent and shuffled primary rays, deterministic secondary rays, and
  optional TinyBVH builder controls.
- Added correctness, primitive-agreement, depth, overflow, fallback, and raw
  timing reporting with deterministic regression fixtures.
- Added the default procedural scene, the hash-pinned CC0 Coastal Cliff 01
  workflow, and optional comparison viewers.
- Archived complete RTX 3050 Ti and paid Lambda A10, A100, and H100 evidence,
  provider billing, and a reproducible cross-GPU comparison chart.
- Added scripted Lambda lifecycle, SSH execution, verified retrieval, and a
  12-hour B200 availability record containing 139 checks and no launch.
- Published the artifact under MIT with REUSE 3.3 metadata, citation metadata,
  third-party boundaries, and reproducibility documentation.

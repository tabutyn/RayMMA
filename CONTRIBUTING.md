# Contributing to RayMMA

RayMMA is a correctness-first performance experiment. Contributions are
welcome when they preserve the controls needed to interpret a result.

## Before opening a pull request

1. Build a Release configuration and run the CPU algebra, GPU WMMA, and Grid
   smoke tests.
2. Keep rays, geometry, BVH builder, converted tree, triangle order, leaf size,
   and timing scope identical across the backends being compared. Identify
   packet organization explicitly; CUDA32 and 16-ray packet modes are
   intentionally different controls.
3. Report correctness before performance. For `validated`, any hit/miss,
   primitive, or depth-tolerance disagreement is a failure to investigate.
   For `uvt-depthsorted` and `e0e1e2`, report false positives, false negatives,
   wrong primitives, invalid outputs, and depth error as part of the result.
   Do not call an approximate harness `PASS` an image-correctness pass.
4. Performance submissions must include CUDA32. The matched CUDA-packet16 path
   (`CUDA16` in historical output) is an arithmetic diagnostic, not the primary
   software-tracing baseline.
5. For performance changes, include the exact command, commit, GPU, driver,
   CUDA toolkit, architecture, warmup count, raw samples, and timing scope.
6. Do not add models, textures, SDK headers, or other third-party material
   without recorded redistribution terms and attribution.

Run the portable core checks with:

```sh
cmake --preset core
cmake --build --preset core --parallel
ctest --preset core
```

Please keep claims narrow. A result on one GPU and one ray distribution should
be described as that result, not as a general architectural conclusion.
Use “FP16-input, FP32-accumulated WMMA” rather than “FP16 Tensor result,” and
distinguish the `validated` hybrid from the no-Möller Tensor-owned variants.

By contributing, you agree that your contribution may be distributed under
the project's MIT License and that you have the right to submit it.

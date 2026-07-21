# Contributing to RayMMA

RayMMA is a correctness-first performance experiment. Contributions are
welcome when they preserve the controls needed to interpret a result.

## Before opening a pull request

1. Build a Release configuration and run the CPU algebra, GPU WMMA, and Grid
   smoke tests.
2. Compare Tensor and matched CUDA on the same rays, tree, triangle order,
   packet width, leaf size, and final FP32 predicate.
3. Report correctness before performance. Include hit/miss differences,
   primitive differences, and maximum depth error. The release gate is strict:
   investigate any equal-depth or coplanar primitive difference explicitly.
4. For performance changes, include the exact command, commit, GPU, driver,
   CUDA toolkit, architecture, warmup count, raw samples, and timing scope.
5. Do not add models, textures, SDK headers, or other third-party material
   without recorded redistribution terms and attribution.

Run the portable core checks with:

```sh
cmake --preset core
cmake --build --preset core --parallel
ctest --preset core
```

Please keep claims narrow. A result on one GPU and one ray distribution should
be described as that result, not as a general architectural conclusion.

By contributing, you agree that your contribution may be distributed under
the project's MIT License and that you have the right to submit it.

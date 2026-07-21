# Changelog

## Unreleased

- Prepared the lab as the standalone RayMMA research artifact.
- Made the headless CUDA benchmark the default build.
- Removed the legacy comparison viewer and its proprietary dependency.
- Added a 32-ray CUDA baseline with independent per-lane traversal.
- Added deterministic first-bounce cosine-weighted diffuse-ray workloads.
- Added optional TinyBVH binned-SAH and spatial-split builder controls.
- Added an uncompressed SAH-split BVH8 control and 4–256 maximum-leaf sweep.
- Added packet-shuffled primary-ray tests and phase-separated measurements.
- Added FP32 validation after the FP16 WMMA candidate filter.
- Fixed four-triangle leaf alignment for odd triangle counts and added
  5/6/7-triangle regressions.
- Routed out-of-range/non-finite FP16 features, coefficients, and results to
  FP32, with explicit fallback counters and an adversarial range regression.
- Strengthened brute-force validation with strict primitive agreement and
  16-by-16 image-coordinate stratification.
- Added raw timing CSV export and an auditable procedural Grid result bundle.
- Documented the candidate-density crossover and the failed FP16 filter.
- Added a textured orbital viewer for the CUDA32, CUDA16, and
  correctness-preserving Tensor16 paths.
- Clarified the MIT licensing boundary and public repository positioning.

# Changelog

## 0.1.0 (unreleased)

- Prepared the lab as the standalone RayMMA research artifact.
- Made the headless CUDA benchmark the default build.
- Removed the legacy comparison viewer and its proprietary dependency.
- Added a 32-ray CUDA baseline with independent per-lane traversal.
- Added deterministic first-bounce cosine-weighted diffuse-ray workloads.
- Added optional TinyBVH binned-SAH and spatial-split builder controls.
- Added an uncompressed SAH-split BVH8 control and 4–256 maximum-leaf sweep.
- Added packet-shuffled primary-ray tests and phase-separated measurements.
- Added FP32 validation after the FP16 WMMA candidate filter.
- Added experimental `uvt-depthsorted` and `e0e1e2` Tensor-owned variants
  using FP16-input, FP32-accumulated WMMA, zero-tolerance edge predicates,
  Tensor-derived closest-depth/BVH clipping, zero Möller checks, and explicit
  approximate-error reporting.
- Added local-depth recovery, per-triangle power-of-two coefficient
  normalization, and nearest-order/cross-leaf clipping fixtures.
- Fixed four-triangle leaf alignment for odd triangle counts and added
  5/6/7-triangle regressions.
- In `validated`, routed out-of-range/non-finite FP16 features, coefficients,
  and results to FP32, with explicit fallback counters and an adversarial range
  regression. The approximate modes deliberately omit that FP32 fallback and
  may turn unsafe inputs into misses.
- Strengthened brute-force validation with strict primitive agreement and
  16-by-16 image-coordinate stratification.
- Added raw timing CSV export and a reproducible CC0 scene result bundle.
- Removed the warp-local compaction experiment and its speculative
  performance framing.
- Added an A100 preset, an archive-producing GPU runner, and Lambda Cloud API
  automation for launch, per-instance SSH rules, retrieval, and termination.
- Documented the candidate-density crossover and the failed FP16 filter.
- Added a textured orbital viewer for CUDA32, matched CUDA-packet16, and the
  reference-validated WMMA path.
- Added the hash-pinned, CC0 Poly Haven Coastal Cliff 01 model and reproducible
  low/mid/high extraction.
- Added REUSE 3.3 metadata, the canonical MIT license text, and a compliance
  workflow.
- Clarified the MIT licensing boundary and public repository positioning.

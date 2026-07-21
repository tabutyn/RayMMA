# RayMMA repository profile

Use this language when creating the standalone public repository. Keep
performance claims tied to the evidence bundle and do not describe RayMMA as a
production ray tracer.

## GitHub description

CUDA/WMMA research benchmark for Tensor Core ray–triangle candidate filtering,
with tuned CUDA baselines, production-BVH controls, correctness tests, and raw
results.

## Suggested topics

`cuda`, `gpu`, `ray-tracing`, `tensor-cores`, `wmma`, `bvh`,
`computer-graphics`, `benchmarking`, `gpgpu`

## One-sentence summary

RayMMA investigates whether dense Tensor Core matrix operations can accelerate
batches of ray–triangle candidate tests while FP32 intersection retains final
correctness.

## Short project summary

RayMMA rearranges ray–triangle intersection into triangle-only coefficients and
ray-only features. One FP16 WMMA tile evaluates 64 ray/triangle candidate pairs,
and FP32 Möller–Trumbore confirms surviving candidates. The repository compares
that method with matched packet traversal and a tuned independent-ray CUDA
baseline using shared BVHs, rays, geometry, and correctness criteria.

The current evidence is deliberately bounded. A narrow crossover appeared for
one shuffled packet workload, but CUDA32 was faster in the broader local quick
sweep. The repository publishes that mixed result, the false-positive
investigation, raw samples, and remaining limitations instead of presenting a
general Tensor Core speedup.

## What makes the repository useful

- The separated WMMA formulation is small enough to study independently.
- Tensor and CUDA paths share traversal inputs and exact final predicates.
- Correctness is checked against matched CUDA and stratified brute force.
- Primary and first-bounce diffuse rays expose coherence sensitivity.
- TinyBVH binned-SAH and spatial-split builders provide established controls.
- Raw samples and failure history make follow-up experiments auditable.

## License wording

RayMMA's original code and documentation are available under the MIT License,
copyright 2026 Thomas Butyn. External dependencies and separately obtained
scene assets retain their own licenses and are not included in that grant.

## Claims to avoid

Do not say that RayMMA beats production ray tracing, proves a general Tensor
Core advantage, is watertight, or is novel. Those conclusions are not
supported by the current single-GPU evidence.

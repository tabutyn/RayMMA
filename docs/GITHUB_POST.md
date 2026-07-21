# The Tensor Core ray-tracing speedup that nearly disappeared

> Draft. The hardened Grid spot check includes raw samples; the larger
> historical Checker/Sponza figures did not retain raw evidence and are
> presented only as the path that motivated the current controls.

I started RayMMA with a simple question: could NVIDIA Tensor Cores accelerate
ray/triangle intersection if the algebra were arranged as dense matrix
multiplication?

The first result looked exciting. On a high-detail scene, my Tensor path
appeared materially faster than a CUDA reference. Then I made the reference
fairer, added a stronger BVH, tested a larger scene, and found six wrong
closest hits caused by an over-aggressive FP16 filter. The headline speedup
nearly disappeared.

That failure became the useful result.

## The method

RayMMA separates the intersection equations into triangle-only coefficients
and ray-only features. One FP16 `m16n16k16` WMMA tile evaluates four values for
every pair in a group of four triangles and sixteen rays: 64 candidate pairs
per tile.

WMMA is only a filter. Candidate pairs are re-evaluated with FP32
Möller–Trumbore, which owns the final hit and closest-depth decision.

The benchmark feeds both Tensor and CUDA paths the same uncompressed
SAH-split BVH8, triangle order, 16-ray packets, rays, leaf size, and final
predicate.

## What survived stronger controls

In the hardened, raw-sample-backed procedural Grid run on an RTX 3050 Ti:

- at a maximum leaf size of 256, Tensor was 3.9% slower for coherent primary
  rays;
- for the same rays deterministically shuffled across packets, Tensor was
  4.7% faster; and
- the separated Tensor leaf stage was slower in both orders.

These kernel times exclude BVH construction, Tensor coefficient/local-frame
packing, allocation, and upload.

So the bounded finding is not “Tensor Cores beat ray tracing.” It is:

> Dense candidate processing can show a narrow WMMA crossover for some packet
> orderings, but the current method does not provide a robust renderer-level
> win.

The next question is whether compaction can assemble dense ray/leaf tiles
without weakening the BVH.

The repository includes the Grid run's raw CUDA-event samples, unmodified
console output, checksums, and sanitized environment metadata in the
[evidence bundle](../results/rtx3050ti-grid-2026-07-19/README.md).

## Why I am publishing the negative parts

The repository includes the false-negative investigation, matched controls,
phase timing, correctness checks, and the limitations that still block a
paper-level claim.

This is currently a one-GPU research artifact. The current harness adds real
first-bounce diffuse rays, an independent-ray CUDA32 control, and optional
TinyBVH SAH/spatial-split controls. On the RTX 3050 Ti quick sweep, CUDA32 was
faster than Tensor in every tested primary and secondary Grid case. The FP16
envelope still has no formal bound, and A100/H100 results remain outstanding.

If you have a different GPU generation or experience with compacted ray work
queues, I would value reproducible result submissions: exact commit, raw
samples, environment, scene hash, and correctness counters included.

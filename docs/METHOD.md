# Method

RayMMA asks a narrow question: when a traversal stage presents many
ray/triangle candidates, can a Tensor Core reject enough pairs to beat the
same work expressed as scalar FP32 CUDA intersections?

It is a hybrid candidate-filtering method, not a replacement for a complete
ray-tracing pipeline.

## Separated intersection algebra

For triangle

```text
P(u,v) = A + u C + v B
```

and ray

```text
R(t) = O + t r,
```

Cramer's rule gives four shared values:

```text
Nt    = dot(A - O, cross(C, B))
Nu    = dot(r, cross(B, A - O))
Nv    = dot(r, cross(A - O, C))
Delta = dot(r, cross(C, B)).
```

The hit coordinates are `t = Nt/Delta`, `u = Nu/Delta`, and
`v = Nv/Delta`. Expanding the triple products separates all triangle-only
terms from all ray-only terms.

For a ray, RayMMA forms the ten-component feature vector

```text
phi = [1, O.x, O.y, O.z, r.x, r.y, r.z, cross(O,r).x,
       cross(O,r).y, cross(O,r).z].
```

Each triangle contributes four coefficient rows, one for each of `Nt`, `Nu`,
`Nv`, and `Delta`. Six zeros pad the inner dimension to 16. The complete
derivation and sign convention are in
[`prototype/ALGEBRA.md`](../prototype/ALGEBRA.md).

## WMMA tile

One `m16n16k16` operation maps:

- four triangles to 16 coefficient rows;
- sixteen rays to 16 feature columns;
- ten useful features plus six zero pads to the inner dimension; and
- four outputs per pair to the 16-by-16 result.

That evaluates the algebra for `4 × 16 = 64` ray/triangle pairs with:

```cpp
wmma::load_matrix_sync(...);
wmma::mma_sync(...);
wmma::store_matrix_sync(...);
```

The inputs are FP16 and accumulation is FP32. Triangle coefficients are
prepacked; ray features are produced per packet.

## End-to-end path

```text
16-ray packet
      |
      v
uncompressed SAH-split BVH8 traversal
      |
      v
candidate leaf (up to configured maximum)
      |
      v
4 triangles × 16 rays FP16 WMMA tiles
      |
      v
permissive barycentric candidate filter
      |
      v
FP32 Möller–Trumbore validation and closest-hit update
```

The builder first creates a binary tree using a 16-bin surface-area heuristic,
then collapses it into eight-wide nodes. Maximum leaf size is a benchmark
variable, not a fixed recommendation; the documented sweep tests maxima from
4 through 256, while actual leaves may be smaller.

Each group of four triangles uses a local center and scale before conversion
to FP16. This reduces overflow and cancellation risk from large world
coordinates; it does not guarantee representability.

## Numerical policy

WMMA never owns the final hit decision:

1. Out-of-range/non-finite ray features or non-finite WMMA results go directly
   to FP32 validation for that ray and four-triangle group.
2. An approximate determinant near zero goes directly to FP32 validation.
3. Approximate `t` is not used as a rejection predicate.
4. Approximate barycentrics use a deliberately wide, two-times-determinant
   envelope.
5. Every surviving pair is re-evaluated with FP32 Möller–Trumbore.
6. FP32 depth determines the closest hit.

The envelope produced zero Tensor-versus-matched-CUDA disagreements in the
recorded suite. It is not backed by a formal FP16 error bound, so it should be
called empirically validated, not mathematically conservative. In source and
output, “exact tests” means the FP32 validation path; it does not mean exact
real arithmetic or a watertight predicate.

An earlier narrower filter rejected six valid Sponza candidates among 147,456
rays. That failure is retained in the results because it explains why
Checker-only validation was insufficient.

## Matched comparison

The Tensor and matched CUDA paths consume the same:

- BVH and triangle order;
- 16-ray packets and ray set;
- leaf-size configuration;
- FP32 intersection predicate and closest-hit output; and
- CUDA-event timing scope.

This isolates the candidate-processing strategy. It does not make the matched
path the fastest possible CUDA renderer: its 32-lane warp owns only 16 rays to
match the WMMA tile.

The stronger CUDA32 control assigns one independent ray to every lane. Each
ray has a private short stack, traverses the same BVH8, and runs the same FP32
intersection predicate. Four 32-ray warps are launched per 128-thread block.
It intentionally does not share a packet traversal stack: doing so charges an
incoherent warp for the union of all nodes touched by its rays and can
manufacture a Tensor advantage.

The optional TinyBVH controls use its established binned-SAH and spatial-split
builders. Their primitive order and hierarchy are converted to RayMMA's common
BVH8 leaf format, so all three intersection kernels consume the identical
converted tree. This tests builder quality, not TinyBVH's compressed GPU
traversal kernel.

The reported integrated scope is the trace kernel only. BVH construction,
Tensor-specific coefficient/local-frame packing, allocation, upload, and
presentation are excluded and must be accounted for in an end-to-end claim.

Secondary mode is a real first-bounce workload. Primary rays are traced,
camera misses are compacted away, and each hit creates a deterministic
cosine-weighted direction in the oriented surface hemisphere. Bounce
generation and compaction time are reported but excluded from trace timing.

## Hypothesis supported by current evidence

In the historical RTX 3050 Ti run, the FP16-WMMA filter beat matched FP32 CUDA
with configured leaf maxima of 128–256 triangles. With selective 4–16 maxima,
WMMA setup, synchronization, inactive pairs, and FP32 fallbacks cost more than
the intersections they avoided. The best fine-leaf CUDA configuration
remained faster in absolute time. Those timings need a release-commit rerun
with raw samples.

The new independent-ray CUDA32 control was faster than Tensor in every local
Grid quick sweep on the RTX 3050 Ti, including primary, diffuse-secondary,
built-in BVH, TinyBVH SAH, and TinyBVH spatial-split cases. This supersedes a
renderer-level interpretation of the earlier half-warp crossover. The next
research question is whether packet/leaf compaction can create dense MMA tiles
without deliberately weakening BVH selectivity.

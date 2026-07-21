# Method

RayMMA asks a narrow question: when a traversal stage presents many
ray/triangle candidates, can a Tensor Core reject enough pairs to beat the
same work expressed as scalar FP32 CUDA intersections?

The default mode is a hybrid candidate-filtering method. Experimental modes
also expose the raw Tensor-owned hit result; none is a replacement for a
complete ray-tracing pipeline.

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

## Default validated path

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
to FP16. Each triangle's four rows then receive one common power-of-two scale,
which preserves the exact `t`, `u`, and `v` ratios before FP16 quantization and
mainly keeps small coefficients out of the subnormal/underflow range. It does
not add relative precision to values already normal in FP16. These transforms
reduce range loss and cancellation risk; they do not guarantee
representability.

## Validated numerical policy

In this mode, WMMA never owns the final hit decision:

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

## Experimental Tensor-owned policies

The `uvt-depthsorted` variant consumes the same four WMMA outputs but removes
the broad envelope and FP32 validation. Let `d`, `U`, and `V` be `Delta`,
`Nu`, and `Nv` oriented by the sign of `Delta`. It requires

```text
U >= 0, V >= 0, U + V <= d.
```

It then recovers local depth from `Nt/Delta` and converts it to the original
ray parameter:

```text
t_world = (Nt / Delta) / frame.scale.
```

The division by `frame.scale` is essential: edge values and `Delta` scale as
`s^2` in a local frame, while `Nt` scales as `s^3`. Accepted depth updates are
written directly to the closest hit, and integrated traversal passes that
depth to every later AABB test. There is no Möller–Trumbore call.

The `e0e1e2` variant packs three oriented edge functions instead of `Nu`,
`Nv`, and a direct determinant row. For vertices `V0=A`, `V1=A+C`, and
`V2=A+B`, it evaluates

```text
E0 = dot(cross(V1,V2),r) + dot(V2-V1,cross(O,r))
E1 = dot(cross(V2,V0),r) + dot(V0-V2,cross(O,r))
E2 = dot(cross(V0,V1),r) + dot(V1-V0,cross(O,r))
Delta = E0 + E1 + E2.
```

The three signed edge values own inclusion; `u=E1/Delta`, `v=E2/Delta`, and
the fourth row supplies `Nt` for the same world-depth calculation. This tests
whether directly evaluating every edge behaves better than deriving the third
edge as `Delta-Nu-Nv`.

Both variants reject an approximate WMMA `|Delta| < 1e-5` after row
normalization and reject non-finite results implicitly through their
comparisons. This is not a world-space, angular, or fully triangle-scale-
invariant threshold; a sufficiently small face-on triangle can still
disappear. `e0e1e2` also forms `Delta` by summing three independently rounded
outputs. An unrepresentable coefficient batch is skipped instead of falling
back to Möller. Timed kernels intentionally omit per-ray FP16 range checks; an
overflowed ray feature will normally turn the quartet into a false miss. The
guard and zero-tolerance edge bounds are empirical policy, not a watertight
bound. They do not constrain `Nt` error. A false positive with underestimated
depth can replace the true surface and prune later BVH nodes, producing an
interior or wrong-surface artifact rather than merely an inaccurate edge.

Accordingly, approximate output reports false positives, false negatives,
wrong primitives, invalid values, and maximum absolute/relative depth error.
Those disagreements do not by themselves fail the research harness. Baseline
correctness, valid indices/finite values, packet capacity, nonempty behavior,
and dedicated nearest-depth and cross-leaf-clipping fixtures remain mandatory.

## Matched comparison

The validated WMMA and matched CUDA-packet16 paths consume the same:

- BVH and triangle order;
- 16-ray packets and ray set;
- leaf-size configuration;
- FP32 intersection predicate and closest-hit output; and
- CUDA-event timing scope.

The approximate variants retain the same BVH, packets, leaf configuration,
and timing scope, but deliberately replace the FP32 predicate and depth with
their Tensor outputs.

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

For phase-separated diagnostics, traversal collects leaves with an infinite
depth bound before either leaf kernel runs. Its displayed sum is therefore a
fixed-work diagnostic, not a decomposition of integrated traversal with
Tensor-owned clipping.

## Interpretation

Algorithm description and measured evidence are kept separate. See
[Findings and evidence](RESULTS.md) for the current measurements and scope.

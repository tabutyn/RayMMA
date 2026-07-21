# Findings and evidence

## Verdict

RayMMA does not currently demonstrate that Tensor Cores are faster for general
ray tracing. It demonstrates a narrower result: mixed-precision WMMA can become
competitive when traversal supplies dense ray/triangle candidate batches, but
a selective BVH often removes so much work that ordinary independent-ray
CUDA32 Möller–Trumbore wins.

The current strongest software baseline is CUDA32. There is no RT Core,
OptiX, Vulkan RT, or DXR comparison in this repository.

## Evidence status

| Evidence | What it supports | What it does not support |
|---|---|---|
| Current source and 15-test core suite (1 CPU, 14 GPU) | Algebra, row layout, world-depth recovery, and regression behavior | Performance on another GPU or formal FP16 robustness |
| [July 21 Coastal Cliff rerun](../results/rtx3050ti-coastal-cliff-2026-07-21/README.md) | Raw current-backend timing and error rates on an openly redistributable scene, with exact source checksums | Other GPUs or production tracing |
| [July 21 public-commit rerun](../results/rtx3050ti-pushed-5e05391-2026-07-21/README.md) | Raw CUDA32, validated-WMMA, leaf, secondary-ray, and builder comparisons on one RTX 3050 Ti | Current normalization, no-Möller variants, other GPUs, or production tracing |
| [July 19 raw Grid bundle](../results/rtx3050ti-grid-2026-07-19/README.md) | Provisional validated-hybrid timing versus matched CUDA-packet16 on one RTX 3050 Ti | CUDA32, selective leaves, secondary rays, current normalization, or no-Möller variants |
| [Historical RTX 3050 Ti notes](RESULTS_RTX3050TI.md) | Candidate-density and packet-order hypotheses | Archival evidence; raw samples, exact source, and scene hashes were not retained |
| July 21 development checks below | Directional behavior of current CUDA32 and no-Möller variants | Citable release evidence; these checks are not tied to a public commit or raw bundle |

The evidence hierarchy matters. The public-commit rerun contains CUDA32 but
uses an older validated-only snapshot. It must not be used to claim anything
about either newer no-Möller mode.

## CC0 replacement-scene rerun

The current worktree was rebuilt with Poly Haven's CC0 Coastal Cliff 01 and
tested at 256x144 with nine retained samples. Blender 4.5.2 produced 8,516,
71,312, and 461,824-triangle tiers. All 15 tests passed.

- At leaf 16, CUDA32 beat validated WMMA in every primary and secondary
  Coastal Cliff comparison.
- At leaf 256, validated WMMA beat same-tree CUDA32 in three of six scene/order
  comparisons, with zero observed correctness disagreement.
- On Coastal Cliff High, coarse validated WMMA was still about `7.8x` coherent
  and `3.3x` shuffled slower than the fastest selective CUDA32 configuration.
- At High/leaf 256, `uvt-depthsorted` crossed same-tree CUDA32 by `1.077x` and
  `1.281x`, but missed 2 of 970 reference hits and chose 3 wrong primitives.
- At High/leaf 256, `e0e1e2` crossed by `1.067x` and `1.289x`, but missed 3
  hits and chose 5 wrong primitives. Its maximum relative depth error was
  `3.28%`.

This strengthens both sides of the result: Tensor-owned intersection can be a
useful fast approximate mode in dense leaves, while selective CUDA32 remains
the fastest exact configuration tested. The [complete bundle](../results/rtx3050ti-coastal-cliff-2026-07-21/README.md)
retains raw CSVs, transcripts, environment, tests, result checksums, and a
manifest identifying the exact tested source.

## Archival rerun: pushed validated-only snapshot

Commit `5e0539121d7b84c247a13a89d88ddeafd5b2fb8f` was rebuilt from a clean clone
and tested with 256×144 rays and nine raw samples per timing scope. It predates
the approximate variants and current normalization.

- CUDA32 beat validated WMMA in all 12 leaf-16 primary comparisons, all 12
  leaf-16 secondary comparisons, and every Grid/TinyBVH builder comparison.
- At deliberately coarse leaf-256 settings, WMMA beat same-tree CUDA32 in five
  of twelve primary comparisons.
- The Checker High leaf-256 crossover was `1.018x` coherent and `1.058x`
  shuffled, but those WMMA times remained `6.3x` and `3.1x` slower than the
  fastest selective-leaf CUDA32 configuration.
- Every recorded correctness comparison passed with zero observed hit,
  primitive, or depth disagreement.

This is the cleanest evidence for the candidate-density thesis: dense WMMA
arithmetic can cross a deliberately weakened same-tree baseline without
producing the fastest tracer. See the [complete archive](../results/rtx3050ti-pushed-5e05391-2026-07-21/README.md)
for raw CSVs, transcripts, environment, input hashes, audit reports, and checksums.

## Provisional archived result: validated versus matched CUDA-packet16

The July 19 Grid bundle used maximum 256-triangle leaves, 256×144 primary rays,
and nine samples:

| Ray order | Matched CUDA-packet16 | Validated hybrid | packet16 / hybrid |
|---|---:|---:|---:|
| coherent | 1.3763 ms | 1.4326 ms | 0.961x |
| packet-shuffled | 2.5702 ms | 2.4546 ms | 1.047x |

The archived console calls the diagnostic `CUDA16`; current output calls it
`CUDA-packet16`. The validated hybrid lost for coherent packets and narrowly
won for shuffled packets. Its isolated Tensor leaf stage was slower in both
orders. The bundle reported no observed reference disagreement, but the tested source belongs to
a pre-public development tree and cannot be fully reconstructed. It is retained
only as a provisional historical record; current claims use the newer bundles.

## Current development observations: no-Möller variants

The following five-sample checks were run on July 21, 2026, on an RTX 3050 Ti
Laptop GPU. They are recorded here to make the present behavior visible, but
they are **not archival evidence**: the tree was still under development, raw
samples were not retained, and Sponza is an external scene.

Coherent 128×72 primary rays:

| Scene / maximum leaf | Variant | CUDA32 | Tensor-owned | CUDA32 / Tensor | Integrated disagreement |
|---|---|---:|---:|---:|---|
| Grid / 16 | `uvt-depthsorted` | 0.3450 ms | 0.5488 ms | 0.629x | 0 FP, 0 FN, 1 wrong primitive / 814 hits |
| Grid / 16 | `e0e1e2` | 0.3461 ms | 0.5489 ms | 0.631x | 0 FP, 0 FN, 0 wrong primitives / 814 hits |
| Grid / 256 | `uvt-depthsorted` | 1.2349 ms | 1.1807 ms | 1.046x | 0 FP, 1 FN, 4 wrong primitives / 814 hits |
| Grid / 256 | `e0e1e2` | 1.2431 ms | 1.1776 ms | 1.056x | 0 FP, 1 FN, 5 wrong primitives / 814 hits |
| Sponza / 16 | `uvt-depthsorted` | 1.1500 ms | 1.4714 ms | 0.782x | 0 FP, 0 FN, 29 wrong primitives / 8,978 hits |
| Sponza / 16 | `e0e1e2` | 1.1500 ms | 1.4540 ms | 0.791x | 0 FP, 0 FN, 48 wrong primitives / 8,978 hits |

Packet shuffling removed every integrated crossover in this check. The two
candidate-rich Grid ratios fell to `0.724x` and `0.723x`; the Sponza ratios
fell to `0.423x` and `0.428x`.

Sponza is the clearest warning against reducing accuracy to a hit/miss count.
Both variants preserved the hit mask in this small image, yet selected wrong
closest primitives and reached about 23% maximum relative depth error. Such an
underestimated depth can also prune a later BVH node, so errors are not
guaranteed to appear only along triangle edges.

The fixed-work Sponza leaf kernel itself was roughly `1.47–1.48x` faster than
matched CUDA-packet16. Integrated traversal was still slower than CUDA32. This
is the central systems result: faster dense candidate arithmetic is not
automatically a faster tracer.

Reproduce the development checks with:

```sh
./build/core/tensor-wide-bvh-bench \
  --quick --scene Grid --variant uvt-depthsorted
./build/core/tensor-wide-bvh-bench \
  --quick --scene Grid --variant e0e1e2
./build/core/tensor-wide-bvh-bench \
  --quick --candidate-rich --scene Grid --variant uvt-depthsorted
./build/core/tensor-wide-bvh-bench \
  --quick --candidate-rich --scene Grid --variant e0e1e2
```

## What the project can conclude

- Candidate density and tile utilization determine whether WMMA overhead can
  be amortized.
- Packet organization changes the balance between shared traversal and dense
  candidate processing.
- A candidate-rich crossover against a matched packet diagnostic can coexist
  with a slower best absolute tracing configuration.
- Removing FP32 validation improves the isolated arithmetic opportunity but
  makes closest-hit accuracy part of the result.
- Directly evaluating three edge functions does not make an FP16-input
  predicate watertight and may add determinant cancellation because three
  independently rounded values are summed.

## What the project cannot conclude

- that Tensor Cores beat RT Cores or production GPU ray tracing;
- that the result generalizes beyond one laptop Ampere GPU;
- that the current approximate modes produce only edge-local errors;
- that the validated FP16 filter is formally conservative;
- that Tensor Cores win a raw `N rays × M triangles` all-pairs sweep; or
- that the coefficient layout is novel.

## Most useful next experiment

The decisive next step is not a larger BVH leaf. It is a fixed-work raw
`N × M` sweep followed by a compacted traversal experiment:

1. measure dense all-pairs CUDA32, CUDA-packet16, validated, and both
   no-Möller modes;
2. report tile occupancy, useful pairs, and packing cost;
3. compact active ray/leaf work from a selective BVH into dense tiles;
4. compare the complete pipeline with independent-ray CUDA32; and
5. archive raw samples from a public commit on several GPU generations.

See [Reproducibility](REPRODUCIBILITY.md) for the timing and correctness
contract required for a new evidence bundle.

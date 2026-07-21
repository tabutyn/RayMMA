# RayMMA

**Do Tensor Cores help ray tracing? Sometimes for dense candidate batches;
not as a general replacement for CUDA ray–triangle intersection in this
implementation.**

[MIT licensed](LICENSE) · C++17/CUDA · NVIDIA WMMA · Reproducible research
artifact

RayMMA maps four triangles by sixteen rays to one `m16n16k16` WMMA operation
with FP16 inputs and FP32 accumulation. It compares that work with ordinary
CUDA Möller–Trumbore intersection inside the same software BVH traversal.

## Short answer

The strongest baseline currently implemented is independent-ray `CUDA32`:
one ray per lane, FP32 Möller–Trumbore, and a selective BVH. On the one tested
RTX 3050 Ti, it was faster than every Tensor path in the local selective-BVH
Grid quick sweep at the default maximum leaf size of 16.

In the [CC0 Coastal Cliff rerun](results/rtx3050ti-coastal-cliff-2026-07-21/README.md),
the no-Möller modes crossed same-tree CUDA32 at `1.06–1.30x` in deliberately
coarse High/leaf-256 work. They missed 2–3 of 970 reference hits and chose
3–5 wrong primitives. They are useful approximate results, not exact wins.

In the same current run, the exact validated hybrid crossed same-tree CUDA32
in three of six coarse-leaf comparisons. On the highest triangle tier it was
still about `9.3x`/`3.3x` slower than the fastest selective CUDA32 settings.

The newer no-Möller variants are useful approximate experiments, but they can
choose the wrong primitive or depth. Their errors are not guaranteed to remain
at silhouettes or shared edges.

## Findings at a glance

- **Selective traversal wins.** Reducing candidate intersections with a good
  BVH was more valuable than accelerating deliberately dense candidate work.
- **Tensor utilization matters.** WMMA becomes more competitive as more of the
  `4 × 16 = 64` ray/triangle pairs are useful.
- **Packet order matters.** Coherent and shuffled rays produced materially
  different traversal costs and crossover sizes.
- **The validated hybrid is empirical.** Its FP16-input WMMA stage is only a
  broad filter and FP32 Möller owns admitted hits. Recorded tests agreed with
  the reference, but there is no formal conservative FP16 bound.
- **Removing Möller exposes real error.** The current archived suite found
  mostly matching hit masks but wrong closest primitives and occasionally
  large relative depth error.
See [Findings and evidence](docs/RESULTS.md) for the complete measured result
and its scope.

## Backends

| Name | Purpose | Final hit owner | Expected accuracy |
|---|---|---|---|
| `CUDA32` | Primary performance baseline; one independent ray per lane | FP32 Möller–Trumbore | Reference software path |
| `CUDA-packet16` | Diagnostic matched to the 16-ray WMMA packet traversal | FP32 Möller–Trumbore | Reference packet path |
| `validated` | FP16-input/FP32-accumulated WMMA broad filter | FP32 Möller–Trumbore | Empirically reference-matching, not proven conservative |
| `uvt-depthsorted` | WMMA `Nt/Nu/Nv/Delta`; direct UV bounds and Tensor-derived depth | Tensor output plus FP32 postprocessing | Intentionally approximate |
| `e0e1e2` | WMMA three direct edges plus `Nt`; summed determinant and Tensor-derived depth | Tensor output plus FP32 postprocessing | Intentionally approximate |

“Tensor-owned” means the WMMA result owns inclusion and depth. It does not
mean the whole tracer runs in FP16: traversal, accumulation, comparisons,
division, and AABB tests still use ordinary CUDA/FP32 operations.

## Quick start

Requirements: CUDA, CMake 3.24+, Python 3, and an NVIDIA GPU with compute
capability 7.0 or newer.

```sh
cmake --preset core
cmake --build --preset core --parallel
ctest --preset core
./build/core/tensor-wide-bvh-bench --quick --scene Grid
```

The default procedural Grid scene is generated in source and needs no
downloaded model, Blender, or TinyBVH.

On a supported single B200, H100, A100, or A10,
`./tools/run_cloud_gpu.sh --profile archive` builds for the native CUDA
architecture, runs the primary and secondary Grid leaf sweeps, and creates a
checksummed result tarball. [Running on Lambda Cloud](docs/LAMBDA_CLOUD.md)
documents the API-only launch, retrieval, verification, and termination path.
The repository retains the complete
[paid Lambda A10 evidence run](results/lambda-a10-2026-07-21/README.md).

For a high-density real mesh with unambiguous redistribution rights, fetch
Poly Haven's CC0 Coastal Cliff 01 and let Blender derive three benchmark tiers:

```sh
./tools/fetch_open_model.sh
cmake --preset core \
  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \
  -DRAYMMA_OPEN_MODEL_ASSET_DIR="$PWD/build/scenes/coastal-cliff-01"
cmake --build --preset core --parallel
./build/core/tensor-wide-bvh-bench --quick --scene CoastalCliffHigh
```

Run the approximate no-Möller variants:

```sh
./build/core/tensor-wide-bvh-bench \
  --quick --scene Grid --variant uvt-depthsorted

./build/core/tensor-wide-bvh-bench \
  --quick --scene Grid --variant e0e1e2
```

Test the candidate-density hypothesis:

```sh
./build/core/tensor-wide-bvh-bench \
  --leaf-sweep --scene Grid --variant validated

./build/core/tensor-wide-bvh-bench \
  --candidate-rich --scene Grid --variant uvt-depthsorted
```

`--candidate-rich` is an alias for a maximum of 256 triangles per leaf. It is
an intentionally coarse control, not a recommended BVH configuration.

The benchmark reports `CUDA32 median / WMMA median`; values below `1.0`
mean CUDA32 is faster. For approximate variants, `PASS` means the harness,
baselines, indices, finite-value checks, and depth regressions passed. It does
**not** mean the rendered result matches the reference. Always read the
false-positive, false-negative, wrong-primitive, and depth-error counters.

See [Reproducing RayMMA](docs/REPRODUCIBILITY.md) for external scenes,
TinyBVH builder controls, secondary rays, raw CSV output, rendering, and the
full timing contract.

## Method in one minute

For triangle `P(u,v) = A + uC + vB` and ray `R(t) = O + tr`, RayMMA separates
Cramer's-rule terms into triangle coefficients and this ray feature vector:

```text
[1 | O.xyz | r.xyz | cross(O,r).xyz | 6 zeros]
```

Four coefficient rows per triangle and sixteen ray columns form one WMMA tile:

```text
4 triangles × 4 outputs  -> 16 rows
16 rays                  -> 16 columns
10 useful features       -> padded inner dimension 16
```

The default `validated` path uses approximate barycentrics only as a permissive
filter. Ambiguous pairs and admitted candidates go to FP32 Möller–Trumbore,
which owns the primitive, barycentrics, and depth.

`uvt-depthsorted` instead applies zero-margin signed UV bounds and uses

```text
t_world = (Nt / Delta) / local_frame_scale
```

for closest-hit ordering and later BVH clipping. `e0e1e2` evaluates all three
oriented edge functions independently, reconstructs `Delta = E0+E1+E2`, and
uses the same depth rule. Neither calls Möller.

Four-triangle local coordinates and a common per-triangle power-of-two row
scale reduce FP16 range loss. They do not make the predicates watertight. The
approximate determinant guard can reject small or grazing geometry, and a bad
underestimated depth can prune later BVH nodes.

The complete derivation and numerical policy are in
[Method](docs/METHOD.md) and [Algebra](prototype/ALGEBRA.md).

## Evidence policy

The published RTX 3050 Ti and paid Lambda A10 bundles cover the current
CUDA32, validated WMMA, `uvt-depthsorted`, and `e0e1e2` backends. They include
raw samples, complete transcripts, environment capture, source hashes, and
all correctness counters. The [results policy](results/README.md) defines what
to retain when adding another GPU or workload.

## Scope and limitations

This repository does **not** currently provide:

- an RT Core, OptiX, Vulkan RT, or DXR comparison;
- a compressed production GPU traversal implementation;
- a raw all-pairs `N rays × M triangles` microbenchmark sweep;
- a formal FP16 error bound or watertight approximate predicate;
- a claim of algorithmic novelty; or
- evidence that Tensor Cores generally accelerate ray tracing.

TinyBVH support supplies established SAH and spatial-split **builders**, then
converts their output to RayMMA's common uncompressed BVH8. It does not use
TinyBVH's production traversal kernel.

## Repository map

- [`src/research_benchmark.cu`](src/research_benchmark.cu) — benchmark,
  backends, accuracy reporting, and procedural fixtures.
- [`src/production_bvh.h`](src/production_bvh.h) and
  [`src/tinybvh_builder.cpp`](src/tinybvh_builder.cpp) — shared BVH format and
  optional builder bridge.
- [`src/live_viewer.cu`](src/live_viewer.cu) and
  [`src/viewer.cpp`](src/viewer.cpp) — optional live and PPM comparison viewers;
  they expose the validated path, not the new approximate variants.
- [`prototype/`](prototype) — minimal tile implementation and algebra tests.
- [`docs/METHOD.md`](docs/METHOD.md) — detailed algorithm and numerical policy.
- [`docs/RESULTS.md`](docs/RESULTS.md) — current findings and evidence status.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — complete commands and
  measurement contract.
- [`docs/LAMBDA_CLOUD.md`](docs/LAMBDA_CLOUD.md) — scripted Lambda Cloud GPU
  launch, benchmark, result retrieval, and termination.
- [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) — related work and novelty
  boundary.
- [`results/`](results) — raw evidence policy and retained bundles.
- [`REUSE.toml`](REUSE.toml) and [`LICENSES/`](LICENSES) — machine-readable
  copyright and license metadata conforming to REUSE 3.3.

## License and citation

Original RayMMA code and documentation are available under the [MIT
License](LICENSE), copyright 2026 Thomas Butyn. CUDA, TinyBVH, separately
obtained scenes, and other dependencies retain their own terms; see
[Third-party notices](THIRD_PARTY.md).

RayMMA is an exploratory research artifact, not a production renderer. If you
use it in research, cite the metadata in [`CITATION.cff`](CITATION.cff) and
report exact commands, raw samples, correctness counters, environment, and
scene hashes.

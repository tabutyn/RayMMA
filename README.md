# RayMMA

**When do Tensor Cores help ray–triangle intersection?**

[MIT licensed](LICENSE) · C++17/CUDA · NVIDIA WMMA · Reproducible benchmark

RayMMA is an open-source C++/CUDA research benchmark for one focused question:
can Tensor Cores help with batches of ray–triangle candidate intersections?
It maps four triangles by sixteen rays to one FP16 `m16n16k16` WMMA tile, then
validates every surviving candidate with FP32 Möller–Trumbore.

## What this repository is

- A compact implementation of separated ray–triangle intersection algebra for
  WMMA.
- A controlled comparison against matched 16-ray CUDA and tuned independent
  32-ray CUDA traversal.
- A benchmark harness with strict correctness tests, raw timing export,
  primary and genuine secondary rays, and optional TinyBVH SAH/SBVH builders.
- A textured orbital viewer for inspecting the same CUDA and Tensor paths on
  user-supplied real geometry.
- A record of both positive and negative results, including an apparent
  speedup that weakened after stronger controls and numerical fixes.

## What this repository is not

- It is not a production renderer or a replacement for hardware ray tracing.
- It does not establish that Tensor Cores are generally faster for ray
  tracing.
- It does not yet establish novelty or a paper-level result.
- It does not redistribute third-party benchmark scenes, Checker assets,
  TinyBVH, or the inspected MIT CUDA ray tracer.

The original RayMMA source and documentation are MIT licensed. Dependencies,
tools, drivers, and separately obtained scenes retain their own terms; see
[THIRD_PARTY.md](THIRD_PARTY.md).

## What the hardened result says

On the redistributable procedural Grid scene, the hardened harness produced a
narrow, ray-order-dependent result:

| RTX 3050 Ti, maximum 256 triangles/leaf | Matched CUDA | Tensor | Speedup |
|---|---:|---:|---:|
| coherent primary rays | 1.3763 ms | 1.4326 ms | 0.961× |
| packet-shuffled primary rays | 2.5702 ms | 2.4546 ms | 1.047× |

These are median integrated trace-kernel times from nine samples. Both orders
passed strict full-image and brute-force correctness checks. The complete
[raw samples, console output, checksums, and environment
metadata](results/rtx3050ti-grid-2026-07-19/README.md) are included.

That is not a broad speedup: Tensor lost on coherent rays and won by 4.7% on
deterministically shuffled primary-ray packets. The separated Tensor leaf
stage was slower in both orders. This bundle does not compare selective leaf
sizes.

## Why the historical crossover is still documented

An earlier Checker High sweep reached 1.21× with a maximum leaf size of 256,
while selective 4–16 maxima were about 5–18% slower. Those measurements did
not retain raw samples and predate the final numerical and BVH-layout
hardening, so they are hypothesis-generating history rather than release
evidence. In that historical sweep, the best selective CUDA configuration was
also faster in absolute time than the coarse Tensor configuration.

![Checker High speedup crosses 1.0 only at larger BVH leaves](docs/checker-high-crossover.svg)

This is a candidate-density crossover against the diagnostic matched
16-ray-packet CUDA path, not evidence that this renderer beats a production
software tracer. The current harness now includes the stronger 32-ray control;
the historical evidence bundle predates it.

That new control materially changes the interpretation. In a July 19 local
quick sweep, the independent-ray CUDA32 path was faster than Tensor for every
Grid primary and secondary case across the built-in, TinyBVH SAH, and TinyBVH
spatial-split builders. Those five-sample spot checks are not a replacement
for an A100/H100 release run, but they rule out treating the earlier
half-warp comparison as a renderer-level speedup.

The most important finding came from a failed result: a narrower FP16 filter
looked 1.31× faster in an isolated leaf test, but a larger Sponza run exposed
six wrong closest hits among 147,456 rays. The wider filter removed those
observed disagreements and also removed the apparent isolated win. That
investigation is documented rather than hidden.

Read the [method](docs/METHOD.md), [full RTX 3050 Ti
history](docs/RESULTS_RTX3050TI.md), and [limitations and related
work](docs/RELATED_WORK.md).

## Quick start

The portable core needs CUDA, CMake 3.24+, Python 3, and a GPU with compute
capability 7.0 or newer:

```sh
cmake --preset core
cmake --build --preset core --parallel
ctest --preset core
./build/core/tensor-wide-bvh-bench --quick --scene Grid
```

The default Grid benchmark does not need Blender or downloaded models. Set
the CUDA architecture explicitly when cross-compiling:

```sh
cmake --preset core -DCMAKE_CUDA_ARCHITECTURES=86
```

If `nvcc` is installed but not on `PATH`, also pass, for example,
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

Useful research modes:

```sh
./build/core/tensor-wide-bvh-bench --scene Grid
./build/core/tensor-wide-bvh-bench --leaf-sweep --scene Grid
./build/core/tensor-wide-bvh-bench --candidate-rich --scene Grid
./build/core/tensor-wide-bvh-bench \
  --ray-mode secondary --scene Grid
./build/core/tensor-wide-bvh-bench \
  --resolution 512x288 --leaf 256 --scene Grid \
  --raw-csv results-grid.csv
./build/core/tensor-wide-bvh-bench \
  --resolution 256x144 --scene Grid \
  --render-prefix raymma-grid
```

`--candidate-rich` means a maximum of 256 triangles per leaf. Primary mode
tests coherent order and a deterministic shuffle. Secondary mode first traces
the camera rays, creates deterministic cosine-weighted diffuse bounces at
actual hit points, compacts misses, and then tests pixel order and a
deterministic shuffle.

`--render-prefix PATH` writes PPM images for CUDA32, diagnostic CUDA16, and
Tensor16. The images use identical unlit procedural albedo so pixel equality
can be checked directly.

### Live High Checker viewer

The optional OpenGL viewer renders an extracted High Checker cache continuously
at 1920×1080 with its albedo texture and an orbital camera:

```sh
cmake -S . -B build/viewer \
  -DRAYMMA_BUILD_VIEWER=ON \
  -DRAYMMA_ENABLE_TINYBVH=ON \
  -DRAYMMA_TINYBVH_ROOT=/path/to/tinybvh
cmake --build build/viewer --target raymma-viewer --parallel
./build/viewer/raymma-viewer \
  /path/to/High1.brtri /path/to/High1-albedo.png 256
```

`Space` cycles CUDA32, CUDA16, and Tensor16. `P` pauses the orbit, and the
arrow keys move the camera.

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for timing scopes,
correctness criteria, external scenes, environment capture, and complete
commands.

## Method in one minute

For each triangle, RayMMA precomputes four coefficient rows. For each ray, it
forms:

```text
[1 | origin.xyz | direction.xyz | cross(origin,direction).xyz | 6 zeros]
```

The matrix product returns the three Cramer's-rule numerators and shared
determinant for 64 ray/triangle pairs. Local coordinate frames reduce FP16
range and cancellation risk.

WMMA is only a permissive candidate filter:

1. near-zero approximate determinants fall back to FP32;
2. non-finite or out-of-range FP16 inputs/results fall back to FP32;
3. approximate depth is not used to reject a pair;
4. a wide barycentric envelope selects candidates; and
5. FP32 intersection owns the final hit and depth order.

The current envelope had zero Tensor-versus-matched-CUDA disagreements in the
tested configurations. It is empirically validated, not formally proven
conservative.

## Optional established BVH builders

TinyBVH 1.8 provides established binned-SAH and spatial-split construction.
RayMMA converts its hierarchy and primitive order to the same uncompressed
BVH8 leaf layout consumed by CUDA32, CUDA16, and Tensor:

```sh
git clone https://github.com/jbikker/tinybvh.git /path/to/tinybvh
cmake -S . -B build/tiny \
  -DRAYMMA_ENABLE_TINYBVH=ON \
  -DRAYMMA_TINYBVH_ROOT=/path/to/tinybvh
cmake --build build/tiny --parallel
./build/tiny/tensor-wide-bvh-bench --quick --bvh-sweep --scene Grid
```

The external header is not vendored. The tested revision and license are in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Optional real geometry

No third-party benchmark scene or Checker asset is included in the MIT source
release. User-supplied scenes are enabled with
`RAYMMA_ENABLE_EXTERNAL_SCENES=ON` and explicit CMake paths.

The included fetch helper requires an explicit acknowledgement because Sibenik
is reported as noncommercial:

```sh
./tools/fetch_benchmark_scenes.sh --accept-separate-scene-licenses
```

Checker L/M/H requires user-supplied `Low1.glb`, `Mid1.glb`, and `High1.glb`
plus Blender. Do not publish those geometry/texture files without confirmed
redistribution rights.

## Tests and controls

The core suite contains:

- 10,000 CPU randomized checks of the separated algebra;
- a standalone GPU WMMA tile test;
- full-image Tensor versus matched-CUDA comparisons;
- full-warp, independent-ray CUDA32 comparisons;
- deterministic first-bounce diffuse rays generated from real primary hits;
- 256 deterministic, 16-by-16 image-stratified brute-force rays;
- strict Tensor primitive agreement, depth tolerance, and overflow checks;
- regression cases for odd leaf counts and FP16 range overflow;
- a tested sweep of 4–256 maximum triangles per leaf;
- integrated and phase-separated timings; and
- workload counters for nodes, leaves, triangle tests, and FP32 fallbacks.

Known gaps are adversarial numerical generators, multiple GPU generations,
a formal FP16 filter bound, compressed production traversal layouts, and
persistent/wavefront scheduling.

## Repository map

- [`src/research_benchmark.cu`](src/research_benchmark.cu): headless benchmark.
- [`src/live_viewer.cu`](src/live_viewer.cu): textured orbital CUDA viewer.
- [`src/tinybvh_builder.cpp`](src/tinybvh_builder.cpp): optional production
  builder bridge.
- [`prototype/`](prototype): minimal WMMA algebra derivation and test.
- [`docs/METHOD.md`](docs/METHOD.md): algorithm and numerical policy.
- [`docs/RESULTS_RTX3050TI.md`](docs/RESULTS_RTX3050TI.md): recorded results.
- [`docs/GITHUB_POST.md`](docs/GITHUB_POST.md): public project-story draft.
- [`docs/REPOSITORY_PROFILE.md`](docs/REPOSITORY_PROFILE.md): copy-ready
  GitHub description, topics, and project summaries.
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md): publication gates.

## Status, citation, and license

RayMMA is an exploratory research artifact, not a production renderer and not
yet evidence of novelty. Reproductions and stronger baselines are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

This lab lives inside a larger private monorepo during development. Do not make
that entire repository public. The allowlisted
`tools/export_public_tree.sh DESTINATION` helper creates a clean RayMMA tree
without Ball Roller history/assets, build output, or third-party SDK headers.

Original RayMMA code and documentation are available under the [MIT
License](LICENSE), copyright 2026 Thomas Butyn. Third-party software, SDKs,
drivers, and scene assets retain their own terms and are excluded from that
grant. Use [`CITATION.cff`](CITATION.cff) to cite the software after the
release URL is finalized.

# Reproducing RayMMA

## Portable core

The default build requires:

- an NVIDIA GPU with compute capability 7.0 or newer;
- a compatible NVIDIA driver and CUDA Toolkit with `nvcc`;
- CMake 3.24 or newer; and
- Python 3 for the CPU algebra test.

From the RayMMA repository root:

```sh
cmake --preset core
cmake --build --preset core --parallel
ctest --preset core
./build/core/tensor-wide-bvh-bench --quick --scene Grid
```

Set the architecture explicitly when building for another machine:

```sh
cmake --preset core -DCMAKE_CUDA_ARCHITECTURES=86
```

If CMake cannot find `nvcc`, pass its absolute path with
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (adjust for the installed
toolkit).

The default procedural Grid path does not require Blender, TinyBVH, or
downloaded scene assets.

## Cloud GPU evidence archive

On a clean checkout running on one supported GPU, the archive helper uses the
`core` preset's native CUDA architecture, captures environment and source
hashes, runs all three Tensor variants over the procedural Grid leaf sweep,
and packages logs, raw CSVs, checksums, and built executables:

```sh
./tools/run_cloud_gpu.sh --profile quick
# or the 256x144 primary + secondary archive suite:
./tools/run_cloud_gpu.sh --profile archive
(cd build && sha256sum -c raymma-cloud-results.tar.gz.sha256)
```

The fixed output paths are `build/raymma-cloud-results.tar.gz` and its
`.sha256` sidecar. A failing test or benchmark still produces a partial
archive with `exit-code.txt` when packaging remains possible. The helper
rejects dirty checkouts, multi-GPU hosts, and GPUs outside the supported set.
The environment capture records the exact model and native architecture, so
the artifact is not presented as evidence from another GPU.

The retained Lambda [A10](../results/lambda-a10-2026-07-21/README.md),
[A100](../results/lambda-a100-2026-07-21/README.md), and
[H100](../results/lambda-h100-2026-07-21/README.md) runs are complete examples
of this archive contract. A 12-hour B200 watch found no capacity, so the
[B200 record](../results/lambda-b200-availability-2026-07-21/README.md) is
availability evidence rather than a benchmark.

Regenerate the published cross-GPU comparison from the original nine-sample
CSVs with:

```sh
python3 tools/plot_cloud_comparison.py \
  --png docs/assets/cloud-gpu-crossover.png
```

The script validates that every plotted GPU/leaf/backend group contains nine
samples, then writes the derived comparison CSV and SVG before optionally
rasterizing the PNG with ImageMagick.

For API-driven rental, SSH, execution, retrieval, and termination, see
[Running on Lambda Cloud](LAMBDA_CLOUD.md).

## Benchmark modes

```sh
./build/core/tensor-wide-bvh-bench
./build/core/tensor-wide-bvh-bench --scene Grid
./build/core/tensor-wide-bvh-bench --leaf 16 --scene Grid
./build/core/tensor-wide-bvh-bench --candidate-rich --scene Grid
./build/core/tensor-wide-bvh-bench \
  --quick --variant uvt-depthsorted --scene Grid
./build/core/tensor-wide-bvh-bench \
  --quick --variant e0e1e2 --scene Grid
./build/core/tensor-wide-bvh-bench --leaf-sweep --scene Grid
./build/core/tensor-wide-bvh-bench --ray-mode secondary --scene Grid
./build/core/tensor-wide-bvh-bench --resolution 512x288 --scene Grid
./build/core/tensor-wide-bvh-bench \
  --resolution 256x144 --scene Grid --render-prefix raymma-grid
./build/core/tensor-wide-bvh-bench \
  --scene Grid --raw-csv results-grid.csv
```

`--candidate-rich` is an alias for a maximum leaf size of 256 triangles.
`--leaf-sweep` tests maxima of 4, 8, 16, 32, 64, 128, and 256. Actual leaves
may be smaller. Primary rays are measured in coherent order and deterministically
shuffled order. `--ray-mode secondary` traces the primaries, emits
cosine-weighted diffuse bounces at actual hits, compacts primary misses, and
measures pixel-ordered and deterministically shuffled bounce rays.

The default `--variant validated` uses FP16-input, FP32-accumulated WMMA output
only as a broad filter and gives the final predicate and depth to FP32
Möller–Trumbore.
`uvt-depthsorted` and `e0e1e2` are approximate Tensor-owned experiments:
FP16-input, FP32-accumulated WMMA edge values own hit inclusion and their
derived `t` owns closest-hit order and integrated BVH clipping. They perform
zero Möller checks; traversal and postprocessing remain ordinary CUDA/FP32.

`--render-prefix raymma-grid` writes `-cuda32.ppm`,
`-cuda-packet16.ppm`, and either `-wmma-validated.ppm`,
`-uvt-depthsorted.ppm`, or `-e0e1e2.ppm`. These images visualize hit and
source-primitive disagreement; same-primitive depth error remains in the
console counters.

## Optional standard scenes

Scene files are not part of the MIT release. Review
[`THIRD_PARTY.md`](../THIRD_PARTY.md) before obtaining them. Then configure
explicit paths:

```sh
cmake --preset core \
  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \
  -DRAYMMA_SIBENIK_PATH=/path/to/sibenik.obj \
  -DRAYMMA_SPONZA_PATH=/path/to/sponza.obj \
  -DRAYMMA_SANMIGUEL_PATH=/path/to/sanmiguel.obj
cmake --build --preset core --parallel
```

Missing optional inputs are skipped. Requesting an unavailable scene by name
returns an error rather than silently substituting another scene.

The `CoastalCliffLow/Mid/High` tiers use Poly Haven's Coastal Cliff 01 model,
which Poly Haven publishes under CC0 1.0. The fetcher pins and verifies the
source glTF, geometry, and 1K textures. Blender 4.5.2 produced 8,516, 71,312,
and 461,824-triangle caches; record the counts and Blender version because
decimation output can vary between Blender releases:

```sh
./tools/fetch_open_model.sh
cmake --preset core \
  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \
  -DRAYMMA_OPEN_MODEL_ASSET_DIR="$PWD/build/scenes/coastal-cliff-01"
```

The downloaded asset may be redistributed under CC0, although RayMMA keeps it
out of the source archive to avoid adding roughly 15 MB. Preserve its identity
and hashes when publishing benchmark results.

## Optional TinyBVH builder controls

Obtain the MIT TinyBVH source separately and configure its path:

```sh
git clone https://github.com/jbikker/tinybvh.git /path/to/tinybvh
git -C /path/to/tinybvh checkout 0e4584287823252cf83f0e9cd072848bec5f79c5
cmake -S . -B build/tiny \
  -DRAYMMA_ENABLE_TINYBVH=ON \
  -DRAYMMA_TINYBVH_ROOT=/path/to/tinybvh
cmake --build build/tiny --parallel
ctest --test-dir build/tiny --output-on-failure
./build/tiny/tensor-wide-bvh-bench --quick --bvh-sweep --scene Grid
```

Available choices are `--bvh builtin`, `--bvh tinybvh-sah`, and
`--bvh tinybvh-sbvh`; `--bvh-sweep` runs all three.

## Timing contract

The research harness:

- performs six untimed warmup rounds;
- collects nine CUDA-event samples by default;
- rotates CUDA32, matched CUDA-packet16, and WMMA launch order;
- reports median, p10, and p90;
- excludes BVH build, allocation, upload, and presentation from integrated
  trace time; and
- separately reports traversal and leaf-processing diagnostics.

The primary speedup is `CUDA32 median / WMMA median`. A value below one
means the WMMA path is slower. The matched CUDA-packet16 ratio remains a
diagnostic for isolating 16-ray packet arithmetic.

Tensor-specific coefficient construction, local-frame construction, packing,
and storage are outside the integrated trace-kernel timing. Include them
separately when evaluating dynamic geometry or end-to-end renderer cost.

The phase-separated traversal uses an infinite ray bound because the closest
hit is not known until its later leaf kernel. Its summed time is diagnostic,
not directly interchangeable with the integrated traversal. Traversal, all
CUDA leaf samples, and all Tensor leaf samples are collected sequentially;
the displayed total is a sum of independently sampled medians, not a median of
paired end-to-end samples. In approximate variants it also omits the candidate
work removed by Tensor-owned depth clipping, so the output labels it
`separated fixed-work`.

## Correctness contract

For `validated`, every full image from CUDA32 and WMMA must match the
CUDA-packet16 hit/miss mask, primitive ID, and depth tolerance. The release
harness also brute-forces up to 256 deterministic, 16-by-16 image-stratified
rays selected by original pixel coordinate in both coherent and packet-shuffled order. A
brute-force primitive mismatch is a failure. Equal-depth or coplanar
differences must be investigated explicitly rather than bypassing the gate.
Packet-leaf overflow is a failure.

For `uvt-depthsorted` and `e0e1e2`, image disagreement is part of the result.
The console separately reports false positives per ray, false negatives and
wrong primitives per reference hit, invalid outputs, and maximum depth error.
A harness `PASS` in these modes means the reference baselines,
bounds/index safety, packet capacity, nonempty-hit sanity check, and dedicated
depth regressions passed; it does **not** mean approximate image equality.
Save the console counters with every timing run.

The suite includes odd 5/6/7-triangle leaf-layout cases, a far-camera
tiny-triangle FP16-range case, a farther-first nearest-hit case, and a
separate-leaf fixture that verifies world depth clips a later BVH leaf. It is
not a proof of numerical safety. Add broader near-edge, near-parallel,
large-coordinate, tiny-triangle, and degenerate generators before making a
formal robustness claim.

## Evidence to save with every result

Run:

```sh
./tools/capture_environment.sh build/core > environment.txt
```

The helper is not an automatic sanitizer. Run it from a clean public checkout
and inspect compiler paths, dirty filenames, and every cache line before
publishing the output.

Archive:

- the exact Git commit and dirty status;
- all raw launch samples, not only copied medians;
- GPU, VRAM, clocks/power mode, driver, CUDA, compiler, and build type;
- command line, resolution, camera, ray construction, and warmup count;
- scene source, license, SHA-256, triangle count, and hit rate;
- leaf size, packet width, tree statistics, tests/ray, and fallback rate; and
- every correctness counter.

`--raw-csv FILE` writes every integrated and phase-separated CUDA-event sample
with scene, ray kind, ray order, BVH builder, Tensor variant, resolution,
maximum leaf size, timing scope, and sample index. Stable Tensor scopes remain
`integrated-tensor` and `separated-tensor-leaves`; use the `tensor_variant`
column to distinguish algorithms. Keep the console output and environment
file beside it; the CSV alone does not record the command, commit, scene hash,
or correctness counters.

The current RTX 3050 Ti bundle under `results/` includes raw samples for
CUDA32, validated WMMA, and both no-Möller variants on the hash-pinned CC0
Coastal Cliff model. See [`RESULTS.md`](RESULTS.md) for the measured result.

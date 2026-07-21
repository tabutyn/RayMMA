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

## Benchmark modes

```sh
./build/core/tensor-wide-bvh-bench
./build/core/tensor-wide-bvh-bench --scene Grid
./build/core/tensor-wide-bvh-bench --leaf 16 --scene Grid
./build/core/tensor-wide-bvh-bench --candidate-rich --scene Grid
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

Checker L/M/H additionally require user-supplied `Low1.glb`, `Mid1.glb`, and
`High1.glb` files plus Blender:

```sh
cmake --preset core \
  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \
  -DRAYMMA_CHECKER_ASSET_DIR=/path/to/licensed/checker-glbs
```

Do not publish those files unless you have documented redistribution rights
for both geometry and embedded textures.

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
- rotates CUDA32, matched CUDA16, and Tensor launch order;
- reports median, p10, and p90;
- excludes BVH build, allocation, upload, and presentation from integrated
  trace time; and
- separately reports traversal and leaf-processing diagnostics.

The primary speedup is `CUDA32 median / Tensor median`. A value below one
means the Tensor path is slower. The matched CUDA16 ratio remains a diagnostic
for isolating 16-ray packet arithmetic.

Tensor-specific coefficient construction, local-frame construction, packing,
and storage are outside the integrated trace-kernel timing. Include them
separately when evaluating dynamic geometry or end-to-end renderer cost.

The phase-separated traversal uses an infinite ray bound because the closest
hit is not known until its later leaf kernel. Its summed time is diagnostic,
not directly interchangeable with the integrated traversal. Traversal, all
CUDA leaf samples, and all Tensor leaf samples are collected sequentially;
the displayed total is a sum of independently sampled medians, not a median of
paired end-to-end samples.

## Correctness contract

For every full image, CUDA32 and Tensor must match CUDA16 hit/miss, primitive
ID, and depth tolerance. The release harness also brute-forces up to 256
deterministic, 16-by-16 image-stratified rays selected by original pixel
coordinate in both
coherent and packet-shuffled order. A brute-force primitive mismatch is a
failure. Equal-depth or coplanar differences must be investigated explicitly
rather than bypassing the gate. Packet-leaf overflow is a failure.

The suite includes odd 5/6/7-triangle leaf-layout cases and a far-camera,
tiny-triangle FP16-range case. It is not a proof of numerical safety. Add
broader near-edge, near-parallel, large-coordinate, tiny-triangle, and
degenerate generators before making a formal robustness claim.

## Evidence to save with every result

Run:

```sh
./tools/capture_environment.sh build/core > environment.txt
```

Archive:

- the exact Git commit and dirty status;
- all raw launch samples, not only copied medians;
- GPU, VRAM, clocks/power mode, driver, CUDA, compiler, and build type;
- command line, resolution, camera, ray construction, and warmup count;
- scene source, license, SHA-256, triangle count, and hit rate;
- leaf size, packet width, tree statistics, tests/ray, and fallback rate; and
- every correctness counter.

`--raw-csv FILE` writes every integrated and phase-separated CUDA-event sample
with scene, ray kind, ray order, BVH builder, resolution, maximum leaf size,
timing scope, and sample index. Keep the console output and environment file
beside it; the CSV alone does not record the command, commit, scene hash, or
correctness counters.

The historical Checker/Sibenik/Sponza RTX 3050 Ti tables are an exploratory
snapshot. They are in
[`RESULTS_RTX3050TI.md`](RESULTS_RTX3050TI.md); their raw samples were not
retained, so they should be rerun before a formal release or paper. The newer
procedural Grid spot check has its own raw bundle under `results/`.

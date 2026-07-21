# Tensor-Core ray/triangle intersection prototype

This package is a testable first implementation of the separated algebra from
`ALGEBRA.md`. It uses the CUDA WMMA calls requested in the design:

```cpp
wmma::load_matrix_sync(...);
wmma::mma_sync(...);
wmma::store_matrix_sync(...);
```

One warp performs one `m16n16k16` multiplication with FP16 operands and FP32
accumulation. It evaluates four triangles against sixteen rays, producing
`Nt`, `Nu`, `Nv`, and the shared `Delta` for every pair.

## Files

- `ALGEBRA.md` — commented derivation, feature layout, and hit predicates.
- `tensor_ray_triangle_wmma.cu` — CUDA kernel plus a standalone GPU test.
- `verify_algebra.py` — dependency-free CPU randomized algebra test.

## First test: algebra only

No GPU or external Python packages are required:

```bash
python3 verify_algebra.py
```

## Build directly with nvcc

Choose the architecture matching the test GPU. Examples include `sm_75` for
Turing, `sm_80`/`sm_86` for Ampere, `sm_89` for Ada, and `sm_90` for Hopper:

```bash
nvcc -std=c++17 -O3 -arch=sm_80 tensor_ray_triangle_wmma.cu \
    -o tensor_ray_triangle_wmma
./tensor_ray_triangle_wmma
```

The code requires compute capability 7.0 or later. Compile for the actual GPU
rather than blindly copying the example architecture.

## Build with the repository CMake project

Run this from the RayMMA repository root:

```bash
cmake --preset core -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build --preset core --target tensor-ray-correctness --parallel
./build/core/tensor-ray-correctness
```

## What the executable verifies

The executable constructs a deterministic `4 triangles x 16 rays` tile. It:

1. precomputes triangle coefficient rows and ray feature columns in FP32;
2. converts both WMMA operands to FP16;
3. executes `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`;
4. compares all 256 raw outputs with an FP32 feature-dot-product reference;
5. compares recovered `(t,u,v)` values with an independent Gaussian solve;
6. exits nonzero if a comparison exceeds its FP16-aware tolerance.

## Important prototype limitations

- This demonstrates the math and tile layout, not a complete BVH traversal.
- Host-side packing is intentionally explicit. Production code should produce
  tiles on the GPU or consume prepacked triangle records.
- The dense tile is economical only when traversal supplies enough useful
  ray/triangle candidates. Empty packet lanes directly waste MMA work.
- FP16 coordinate range is the main correctness risk. Convert geometry and ray
  origins to a local BVH-leaf coordinate frame before production benchmarking.
- Boundary and nearly parallel cases may require an FP32 fallback for watertight
  behavior.
- Benchmark against an optimized CUDA intersection kernel and the target GPU's
  RT hardware. A successful arithmetic test does not establish a speedup.

## Where the full experiment lives

This prototype is intentionally frozen as the smallest test of the separated
tile algebra. The BVH-integrated benchmark, local coordinate frames,
Tensor-owned depth variants, CUDA32 control, timing scopes, and accuracy
reporting are implemented in
[`../src/research_benchmark.cu`](../src/research_benchmark.cu).

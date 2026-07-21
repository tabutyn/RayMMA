# RTX 3050 Ti archival rerun of pushed commit 5e05391

This bundle records a rerun of the exact commit published at
[`tabutyn/RayMMA`](https://github.com/tabutyn/RayMMA) on July 21, 2026:

```text
5e0539121d7b84c247a13a89d88ddeafd5b2fb8f
testing tensor cores with raytracing
```

The remote `main` ref was verified with `git ls-remote` before building. The
build used a fresh local clone, Release mode, CUDA architecture 86, CUDA 13.1,
and an RTX 3050 Ti Laptop GPU. All eleven tests passed.

## Important source limitation

The pushed commit is the older validated-hybrid snapshot. It contains CUDA32,
the matched 16-ray CUDA diagnostic (`CUDA16` in this revision), and an
FP16-input/FP32-accumulated WMMA filter followed by FP32 Möller–Trumbore. It
does **not** contain `uvt-depthsorted`, `e0e1e2`, Tensor-owned depth clipping,
the newer normalization, or their depth-order/clipping regressions.

Consequently, this is archival evidence only for the validated hybrid in
commit `5e05391`. It must not be used as evidence for the newer no-Möller
variants in the development tree.

## Honest result

At the selective maximum leaf size of 16, CUDA32 beat validated WMMA in all
12 primary scene/order comparisons. The WMMA path also lost every secondary
ray comparison and every Grid/TinyBVH builder comparison.

At the deliberately coarse maximum leaf size of 256, validated WMMA beat
CUDA32 in five of twelve primary comparisons. Those are same-tree crossovers,
not best-configuration wins. Checker High illustrates the distinction:

- leaf 256: WMMA beat same-tree CUDA32 by `1.018x` coherent and `1.058x`
  shuffled;
- best selective CUDA32: `0.4086 ms` coherent and `0.6441 ms` shuffled;
- leaf-256 WMMA: `2.5762 ms` coherent and `2.0275 ms` shuffled.

Thus the crossover required enough extra candidate work that WMMA remained
about `6.3x` and `3.1x` slower than the fastest measured CUDA32 configuration.
This supports the candidate-density hypothesis while strengthening the main
negative result: BVH selectivity is more valuable than accelerating dense
candidate arithmetic in this implementation.

Every recorded configuration reported zero hit/miss, primitive, or depth
disagreement and passed its brute-force sample. This is empirical agreement,
not a formal conservative FP16 bound.

## Primary, maximum leaf 16

Medians are milliseconds. Ratio is `CUDA32 / WMMA`; below one means CUDA32 is
faster.

| Scene | Coherent CUDA32 | Coherent WMMA | Ratio | Shuffled CUDA32 | Shuffled WMMA | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Grid | 0.4516 | 0.7352 | 0.614x | 0.6072 | 1.4715 | 0.413x |
| Checker Low | 0.2488 | 0.3779 | 0.659x | 0.4035 | 0.8989 | 0.449x |
| Checker Mid | 0.3788 | 0.7209 | 0.525x | 0.5571 | 1.2861 | 0.433x |
| Checker High | 0.4956 | 1.2073 | 0.411x | 0.6400 | 1.6433 | 0.389x |
| Sibenik | 2.2618 | 2.4709 | 0.915x | 5.7498 | 11.5322 | 0.499x |
| Sponza | 2.8754 | 3.3638 | 0.855x | 7.6052 | 17.2390 | 0.441x |

## Primary, maximum leaf 256

| Scene | Coherent CUDA32 | Coherent WMMA | Ratio | Shuffled CUDA32 | Shuffled WMMA | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Grid | 1.4213 | 1.4438 | 0.984x | 2.1206 | 2.9891 | 0.709x |
| Checker Low | 0.5282 | 0.5591 | 0.945x | 1.1334 | 1.5749 | 0.720x |
| Checker Mid | 1.4612 | 1.2769 | 1.144x | 1.8893 | 1.9080 | 0.990x |
| Checker High | 2.6122 | 2.5436 | 1.027x | 2.1402 | 2.0265 | 1.056x |
| Sibenik | 4.0999 | 3.0730 | 1.334x | 9.8571 | 15.4604 | 0.638x |
| Sponza | 5.5931 | 4.7841 | 1.169x | 15.1006 | 24.8932 | 0.607x |

Grid still did not beat CUDA32 at leaf 256. Its shuffled `1.055x` result was
only against the matched CUDA16 packet diagnostic; against CUDA32 it was
`0.709x`.

## Additional controls

- Checker High crossed CUDA32 only at maximum leaf 256. At leaves 4 through
  128, `CUDA32 / WMMA` ranged from `0.284x` to `0.847x`.
- All twelve secondary-ray ratios at leaf 16 were below one, ranging from
  `0.421x` to `0.537x` against CUDA32.
- With TinyBVH binned-SAH and spatial-split Grid trees, ratios were
  `0.355–0.364x`; CUDA32 remained substantially faster.
- The TinyBVH checkout was pinned to
  `0e4584287823252cf83f0e9cd072848bec5f79c5`.

## Commands

The build used the separately licensed inputs identified below and did not
copy them into this bundle:

```sh
cmake -S . -B build/archive \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DRAYMMA_ENABLE_TINYBVH=ON \
  -DRAYMMA_TINYBVH_ROOT=/path/to/tinybvh-0e458428 \
  -DRAYMMA_ENABLE_EXTERNAL_SCENES=ON \
  -DRAYMMA_CHECKER_ASSET_DIR=/path/to/checker-glbs \
  -DRAYMMA_SIBENIK_PATH=/path/to/sibenik.obj \
  -DRAYMMA_SPONZA_PATH=/path/to/sponza.obj
cmake --build build/archive --parallel
ctest --test-dir build/archive --output-on-failure

./build/archive/tensor-wide-bvh-bench \
  --raw-csv primary-leaf16.csv
./build/archive/tensor-wide-bvh-bench --candidate-rich \
  --raw-csv primary-leaf256.csv
./build/archive/tensor-wide-bvh-bench --leaf-sweep --scene CheckerHigh \
  --raw-csv checker-high-leaf-sweep.csv
./build/archive/tensor-wide-bvh-bench --ray-mode secondary \
  --raw-csv secondary-leaf16.csv
./build/archive/tensor-wide-bvh-bench --bvh-sweep --scene Grid \
  --raw-csv grid-bvh-sweep.csv
```

All benchmark commands used the default 256×144 resolution, six untimed
warmup rounds, nine timed samples, and rotating backend order.

## Input identity and licensing

Scene files are not redistributed here.

| Input | SHA-256 | Reported terms |
|---|---|---|
| Procedural Grid | generated by commit `5e05391` | RayMMA MIT |
| Checker Low GLB | `f84b9d8d340a45d3625ed257546f2e2e359476e58dddb9300ca33ebced157aad` | historical external input; not included |
| Checker Mid GLB | `279252981d789b68b8935de8da376801f24d3db9a607ae8316b0997b2ea4d700` | historical external input; not included |
| Checker High GLB | `c0eb20f9302a078c88c5187e89556bab4d0dd841721c771be423bdbb162853b4` | historical external input; not included |
| Sibenik OBJ | `40494f9fa83771e3c4ea4442399042b2fa098d10206db6a6d6da7e48e2182289` | CC BY-NC; external input not included |
| Sponza OBJ | `eee3e272e2c3fc6ab5b7a3868191e07ff7e363af88d3352747112fe58a8c36d4` | CC BY 3.0; not redistributed here |

## Secret and license audit

Gitleaks `8.30.1` was downloaded from its official GitHub release and verified
against the published SHA-256
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
It scanned both reachable commits, about 225 KB, and found zero secrets. A
working-checkout scan also found zero secrets; it skipped a local generated
52 MB Checker triangle cache that is not in Git history or this bundle.

REUSE `6.2.0` found no bad, deprecated, or invalid SPDX license expression,
but the pushed tree is not REUSE 3.3 compliant:

- only 12 of 34 scanned files contain machine-readable license information;
- only 2 of 34 contain recognized copyright information;
- the project uses inline SPDX MIT identifiers without the REUSE-required
  `LICENSES/MIT.txt` copy; and
- many documentation and result files have no per-file SPDX metadata.

The root `LICENSE` is a conventional complete MIT grant, so this is a
machine-readable compliance gap rather than evidence of an incompatible
license. The pushed commit also contains a gitlink at
`third_party/mit-cuda-raytracer` without a `.gitmodules` entry. It points to
MIT-licensed upstream commit `2edc542b84cce3d264e09d5cdf71afe5c3d95a98`,
but the dangling gitlink should be removed or made into a valid submodule.

## Files

- `environment.txt`: clean source identity, GPU, driver, clocks, and toolchain.
- `ctest.txt`: complete 11-test transcript.
- `*.csv`: every CUDA-event timing sample.
- matching `*.txt`: unmodified benchmark console output.
- `gitleaks-*.json`: redacted scanner reports; both are empty arrays.
- `reuse-lint.json`: complete REUSE result for the clean pushed tree.
- `SHA256SUMS`: hashes for the archive contents.

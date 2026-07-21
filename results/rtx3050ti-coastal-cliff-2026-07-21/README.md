# RTX 3050 Ti Coastal Cliff rerun

This bundle validates the current RayMMA worktree against the openly
redistributable Coastal Cliff 01 replacement scene. `SOURCE_SHA256SUMS`
identifies every benchmark, correctness, build, and extraction source used by
the run. `environment.txt` records the complete build environment and worktree
state.

## Environment and method

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB
- Driver: 590.44.01; CUDA compiler/runtime: 13.1
- Build: Release, compute capability 8.6
- Resolution: 256x144
- Timing: 6 warmups followed by 9 CUDA-event samples
- Packet orders: coherent and packet-shuffled
- Builder: RayMMA built-in binned SAH to uncompressed BVH8
- Blender: 4.5.2 LTS
- Scene tiers: 8,516 / 71,312 / 461,824 triangles
- Source asset: Poly Haven Coastal Cliff 01, CC0 1.0

The source glTF package is fetched and SHA-256 verified by
`tools/fetch_open_model.sh`; exact component hashes and attribution are in
`THIRD_PARTY.md`. The model itself is not copied into this bundle.

## Result

All 15 tests passed. Every CUDA32 and validated-WMMA benchmark reported zero
false hits, missed hits, wrong primitives, invalid results, and depth
disagreements.

At selective 16-triangle leaves, CUDA32 beat validated WMMA for every primary
and secondary Coastal Cliff comparison. At deliberately coarse 256-triangle
leaves, validated WMMA beat same-tree CUDA32 in three of six Coastal Cliff
scene/order comparisons:

| Scene | Order | CUDA32 ms | Validated WMMA ms | CUDA32 / WMMA |
|---|---|---:|---:|---:|
| Low | coherent | 0.6963 | 0.7313 | 0.952x |
| Low | shuffled | 0.6124 | 0.6359 | 0.963x |
| Mid | coherent | 1.7275 | 1.7183 | 1.005x |
| Mid | shuffled | 1.0865 | 0.9667 | 1.124x |
| High | coherent | 2.8160 | 3.0607 | 0.920x |
| High | shuffled | 1.1448 | 0.9984 | 1.147x |

This is a same-tree candidate-density crossover, not a best-renderer win. In
the High leaf sweep, the fastest CUDA32 medians were 0.3942 ms coherent at
leaf 4 and 0.3052 ms shuffled at leaf 8. The coarse validated-WMMA results were
therefore about 7.8x and 3.3x slower than the best selective CUDA32 settings.

The no-Moller modes crossed same-tree CUDA32 at High/leaf 256, with explicitly
approximate output:

| Variant | Order | CUDA32 / WMMA | Misses | Wrong primitive | Max relative depth error |
|---|---|---:|---:|---:|---:|
| `uvt-depthsorted` | coherent | 1.077x | 2 | 3 | 1.65% |
| `uvt-depthsorted` | shuffled | 1.281x | 2 | 3 | 1.65% |
| `e0e1e2` | coherent | 1.067x | 3 | 5 | 3.28% |
| `e0e1e2` | shuffled | 1.289x | 3 | 5 | 3.28% |

Those misses are 0.21% and 0.31% of the 970 reference hits. Neither mode
produced a false positive in this run. They remain approximate alternatives,
not replacements for validated Moller-Trumbore when exact output is required.

## Commands

```sh
ctest --test-dir build/open-model-check --output-on-failure

./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 16 --variant validated --raw-csv validated-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 256 --variant validated --raw-csv validated-leaf256.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf-sweep --scene CoastalCliffHigh --variant validated \
  --raw-csv high-leaf-sweep.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 16 --ray-mode secondary --variant validated \
  --raw-csv validated-secondary-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 16 --variant uvt-depthsorted --raw-csv uvt-depthsorted-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 16 --variant e0e1e2 --raw-csv e0e1e2-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 256 --scene CoastalCliffHigh --variant uvt-depthsorted \
  --raw-csv uvt-depthsorted-high-leaf256.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 256 --scene CoastalCliffHigh --variant e0e1e2 \
  --raw-csv e0e1e2-high-leaf256.csv
```

Each `.txt` file is the corresponding complete console transcript. Each CSV
contains all raw timing samples; no rows were removed as outliers.

Verify the tested source from the repository root with:

```sh
sha256sum --check \
  results/rtx3050ti-coastal-cliff-2026-07-21/SOURCE_SHA256SUMS
```

# RTX 3050 Ti Coastal Cliff reproducible benchmark

This bundle records the current RayMMA source against the hash-pinned, CC0
Poly Haven Coastal Cliff 01 model. `SOURCE_SHA256SUMS` identifies the tested
benchmark, correctness, build, and extraction source.

## Environment and method

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB
- Driver: 590.44.01; CUDA compiler/runtime: 13.1
- Build: Release, compute capability 8.6
- Resolution: 256x144
- Timing: 6 internal warmups followed by 9 retained CUDA-event samples
- GPU preconditioning: one quick candidate-rich Grid run immediately before
  every recorded benchmark process
- Packet orders: coherent and packet-shuffled
- Builder: built-in binned SAH converted to uncompressed BVH8
- Blender: 4.5.2 LTS
- Scene tiers: 8,516 / 71,312 / 461,824 triangles
- Source asset: Poly Haven Coastal Cliff 01, CC0 1.0

`tools/fetch_open_model.sh` downloads and verifies the source glTF package.
Exact component hashes and attribution are recorded in `THIRD_PARTY.md`.

## Result

All 15 tests passed. CUDA32 and validated WMMA reported zero false hits,
missed hits, wrong primitives, invalid results, and depth disagreements.

At selective 16-triangle leaves, CUDA32 beat validated WMMA in every primary
and secondary Coastal Cliff comparison. At deliberately dense 256-triangle
leaves, validated WMMA beat same-tree CUDA32 in three of six primary
comparisons:

| Scene | Order | CUDA32 | Validated WMMA | CUDA32 / WMMA |
|---|---|---:|---:|---:|
| Low | coherent | 0.6963 ms | 0.7373 ms | 0.944x |
| Low | shuffled | 0.6060 ms | 0.6255 ms | 0.969x |
| Mid | coherent | 1.7406 ms | 1.7080 ms | 1.019x |
| Mid | shuffled | 1.0854 ms | 0.9532 ms | 1.139x |
| High | coherent | 2.8140 ms | 3.0515 ms | 0.922x |
| High | shuffled | 1.1387 ms | 1.0043 ms | 1.134x |

The High leaf sweep found the fastest CUDA32 medians at 0.3267 ms coherent
and 0.3031 ms shuffled. The leaf-256 validated result was therefore about
9.3x and 3.3x slower than the best selective CUDA32 configuration.

The no-Möller modes crossed same-tree CUDA32 at High/leaf 256 with approximate
output:

| Variant | Order | CUDA32 / Tensor | Misses | Wrong primitive | Max relative depth error |
|---|---|---:|---:|---:|---:|
| `uvt-depthsorted` | coherent | 1.075x | 2 | 3 | 1.65% |
| `uvt-depthsorted` | shuffled | 1.292x | 2 | 3 | 1.65% |
| `e0e1e2` | coherent | 1.064x | 3 | 5 | 3.28% |
| `e0e1e2` | shuffled | 1.299x | 3 | 5 | 3.28% |

The misses are 0.21% and 0.31% of 970 reference hits. Neither mode produced a
false positive in this run.

## Commands

Before each recorded benchmark command, the GPU was preconditioned with:

```sh
./build/open-model-check/tensor-wide-bvh-bench \
  --quick --candidate-rich --scene Grid --variant validated
```

The recorded commands were:

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
  --leaf 16 --variant uvt-depthsorted \
  --raw-csv uvt-depthsorted-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 16 --variant e0e1e2 --raw-csv e0e1e2-leaf16.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 256 --scene CoastalCliffHigh --variant uvt-depthsorted \
  --raw-csv uvt-depthsorted-high-leaf256.csv
./build/open-model-check/tensor-wide-bvh-bench \
  --leaf 256 --scene CoastalCliffHigh --variant e0e1e2 \
  --raw-csv e0e1e2-high-leaf256.csv
```

Each CSV contains all retained samples. Each `.txt` file is the complete
corresponding console transcript.

Verify the archive and tested source from the repository root:

```sh
(cd results/rtx3050ti-coastal-cliff-2026-07-21 && \
  sha256sum --check SHA256SUMS)
sha256sum --check \
  results/rtx3050ti-coastal-cliff-2026-07-21/SOURCE_SHA256SUMS
```

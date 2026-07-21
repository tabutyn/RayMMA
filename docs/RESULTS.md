# Findings and evidence

## Verdict

RayMMA does not show that Tensor Cores are generally faster for ray tracing.
It shows a narrower systems result: dense ray/triangle candidate batches can
benefit from WMMA, while a selective BVH plus ordinary CUDA32
Möller–Trumbore remains the fastest complete configuration tested.

## Reproduced result

The current standalone source was rebuilt with CUDA 13.1 and the hash-pinned,
CC0 Poly Haven Coastal Cliff 01 model. The model produced 8,516, 71,312, and
461,824-triangle tiers. All 15 tests passed.

The benchmark used 256x144 rays, six internal warmups, nine retained
CUDA-event samples, and a separate candidate-rich Grid run immediately before
each recorded process to bring the laptop GPU out of its idle clock state.

At a selective maximum leaf size of 16, CUDA32 beat the validated and
Tensor-owned paths in every Coastal Cliff primary comparison. CUDA32 also beat
the validated path for every measured secondary-ray comparison.

At a deliberately dense maximum leaf size of 256, the exact validated hybrid
beat same-tree CUDA32 in three of six comparisons:

| Scene | Order | CUDA32 | Validated WMMA | CUDA32 / WMMA |
|---|---|---:|---:|---:|
| Low | coherent | 0.6963 ms | 0.7373 ms | 0.944x |
| Low | shuffled | 0.6060 ms | 0.6255 ms | 0.969x |
| Mid | coherent | 1.7406 ms | 1.7080 ms | 1.019x |
| Mid | shuffled | 1.0854 ms | 0.9532 ms | 1.139x |
| High | coherent | 2.8140 ms | 3.0515 ms | 0.922x |
| High | shuffled | 1.1387 ms | 1.0043 ms | 1.134x |

The High leaf sweep found the fastest CUDA32 medians at 0.3267 ms coherent
and 0.3031 ms shuffled. The leaf-256 validated path was therefore about 9.3x
and 3.3x slower than the best selective CUDA32 configuration despite one
same-tree crossover.

The Tensor-owned modes remove every Möller check and use Tensor-derived depth
for closest-hit ordering and BVH clipping. On Coastal Cliff High at leaf 256:

| Variant | Order | CUDA32 / Tensor | Misses | Wrong primitive | Max relative depth error |
|---|---|---:|---:|---:|---:|
| `uvt-depthsorted` | coherent | 1.075x | 2 | 3 | 1.65% |
| `uvt-depthsorted` | shuffled | 1.292x | 2 | 3 | 1.65% |
| `e0e1e2` | coherent | 1.064x | 3 | 5 | 3.28% |
| `e0e1e2` | shuffled | 1.299x | 3 | 5 | 3.28% |

These are approximate throughput wins against the same dense tree, not exact
best-renderer wins. The complete
[benchmark bundle](../results/rtx3050ti-coastal-cliff-2026-07-21/README.md)
contains the raw samples, full transcripts, environment, tests, checksums, and
source manifest.

## What this means

- BVH selectivity saved more work than WMMA accelerated in the best complete
  configuration.
- Dense candidate work can amortize Tensor Core setup and make both exact
  hybrid and approximate Tensor-owned kernels competitive with the same-tree
  CUDA path.
- Removing FP32 Möller validation improves throughput but makes hit, primitive,
  and depth error part of the result.
- Packet order materially changes traversal and batching costs.

## Scope

This is a software BVH comparison on one RTX 3050 Ti Laptop GPU. The strongest
implemented baseline is independent-ray CUDA32; the repository does not
compare against RT Cores, OptiX, Vulkan RT, or DXR. Results on other GPU
generations remain useful future measurements.

See [Reproducibility](REPRODUCIBILITY.md) for the timing and correctness
contract.

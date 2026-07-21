# Lambda Cloud A100 archive benchmark

This bundle records a paid Lambda Cloud run on July 21, 2026. It was built
from public commit `f6e767fe53f28abd2842ce994875a8c2ca0e1d98` on one NVIDIA
A100-SXM4-40GB and completed the full `archive` profile. The archive was
downloaded and checksum-verified before the Lambda API confirmed termination.
A separate post-run inventory query reported no running instances.

The live launch price was $1.99/hour. The helper measured 10.2 minutes from
launch through confirmed termination and estimated $0.36 of compute before
tax. Lambda's invoice remains the authoritative billing record.

## Environment and test

- Lambda instance: `gpu_1x_a100_sxm4` in `us-east-1`
- Lambda image: `lambda-stack-24-04`
- GPU: NVIDIA A100-SXM4-40GB, compute capability 8.0
- Driver: 580.105.08
- CUDA compiler/runtime: 12.8.93
- CMake: 3.28.3
- Build: Release, native CUDA architecture
- Scene: procedural Grid, 32,768 triangles
- Resolution: 256×144
- Samples: six warmups and nine retained CUDA-event samples
- Leaf maxima: 4, 8, 16, 32, 64, 128, and 256 triangles
- Rays: coherent/shuffled primary and pixel-ordered/shuffled deterministic
  secondary rays
- Variants: `validated`, `uvt-depthsorted`, and `e0e1e2`
- CTest: 16/16 passed
- Remote runner exit code: 0
- Stack and packet overflow counters: zero throughout the archive run

`git-status.txt` contains only `?? build/`, created after the clean-checkout
guard by the build itself. The source commit was detached and unchanged.

## Performance result

Speedup is median integrated CUDA32 time divided by median integrated Tensor
time. Above 1.0 means the Tensor path was faster.

The exact `validated` hybrid crossed CUDA32 only for coherent primary rays at
leaf 128, by 1.024×. It remained slower for shuffled primary rays and every
secondary-ray case. This narrow crossover does not change the main result:
selective BVH traversal plus CUDA32 Möller–Trumbore was the fastest complete
configuration.

The no-Möller approximate paths crossed over more strongly at dense leaves:

| Workload | Best `uvt-depthsorted` | Best `e0e1e2` |
|---|---:|---:|
| Primary, coherent | 1.716× at leaf 128 | 1.709× at leaf 128 |
| Primary, shuffled | 1.168× at leaf 128 | 1.170× at leaf 128 |
| Secondary, pixel-ordered | 1.191× at leaf 64 | 1.193× at leaf 128 |
| Secondary, shuffled | 1.126× at leaf 128 | 1.127× at leaf 128 |

These are same-tree dense-candidate comparisons, not claims against an
optimally selective renderer or hardware RT cores.

## Accuracy result

The validated path matched the reference throughout the archive. Maximum
disagreement for the Tensor-owned variants was:

| Variant and ray set | Max false positives/rays | Max false negatives/reference hits | Max wrong primitive/reference hits |
|---|---:|---:|---:|
| `uvt-depthsorted`, primary | 0.0027% | 0.2452% | 0.2758% |
| `e0e1e2`, primary | 0.0027% | 0.1839% | 0.3065% |
| `uvt-depthsorted`, secondary | 0.0000% | 0.4710% | 0.4710% |
| `e0e1e2`, secondary | 0.0000% | 0.1570% | 0.7849% |

## Evidence map

- `environment.txt`, `commit.txt`, `source-sha256.txt`, and
  `git-status.txt`: machine and source provenance.
- `configure.log`, `build.log`, `tests.log`, and `commands.txt`: exact build
  and test record.
- `grid-*.csv`: every retained timing sample.
- `grid-*.log`: complete performance, work, and correctness counters.
- `SHA256SUMS`: checksums for every file inside the original archive.
- `raymma-cloud-results.tar.gz` and its `.sha256` sidecar: the original
  locally verified download.
- `bin/`: the SM80 executables built on the rental; rebuild from source rather
  than treating them as portable binaries.

The procedural Grid requires no third-party geometry or textures.

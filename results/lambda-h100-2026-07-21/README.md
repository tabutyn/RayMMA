# Lambda Cloud H100 archive benchmark

This bundle is the evidence from a paid Lambda Cloud run on July 21, 2026.
It was built from public commit
`07df9d82e7bc4c15a27eb12be9cea318872e8093` on one NVIDIA H100 80GB HBM3
and completed the full `archive` profile. The archive was downloaded and
checksum-verified before the Lambda API confirmed termination. A separate
post-run inventory query reported no running instances.

The live launch price was $4.29/hour. The helper measured 7.4 minutes from
launch through confirmed termination and estimated $0.57 of compute before
tax. Lambda's invoice remains the authoritative billing record.

## Environment and test

- Lambda instance: `gpu_1x_h100_sxm5` in `us-south-2`
- Lambda image: `lambda-stack-24-04`
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
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

The exact `validated` hybrid did not beat CUDA32 in any integrated H100 case.
Its best results were 0.968× for coherent primary rays, 0.947× for shuffled
primary rays, 0.791× for pixel-ordered secondary rays, and 0.781× for shuffled
secondary rays. The final FP32 Möller–Trumbore validation therefore preserved
the reference result but did not produce an end-to-end speedup.

The no-Möller approximate paths crossed over when candidate leaves became
dense:

| Workload | Best `uvt-depthsorted` | Best `e0e1e2` |
|---|---:|---:|
| Primary, coherent | 1.586× at leaf 128 | 1.590× at leaf 128 |
| Primary, shuffled | 1.191× at leaf 64 | 1.190× at leaf 64 |
| Secondary, pixel-ordered | 1.109× at leaf 128 | 1.124× at leaf 128 |
| Secondary, shuffled | 1.078× at leaf 128 | 1.079× at leaf 128 |

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
- `bin/`: the two SM90 executables built on the rental; rebuild from source
  instead of treating them as portable binaries.

The procedural Grid requires no third-party geometry or textures.

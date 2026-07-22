# Lambda Cloud A10 archive benchmark

This bundle is the evidence from a paid Lambda Cloud run on July 21, 2026.
It was built from public commit
`0c89dd87c021a20c7019e74f9e2b373b66661157` on a single NVIDIA A10 and
completed the full `archive` profile. The orchestration client downloaded and
verified the archive before requesting termination; Lambda then reported the
instance `terminated`, and a separate inventory query reported no running
instances.

The live launch price was $1.29/hour. The helper measured 9.3 minutes from
its launch request through confirmed termination and conservatively estimated
$0.21 by applying that price to the whole wall-clock lifecycle. Lambda's
billing history later reported 20:28–20:32 UTC, displayed 0.07 hours, and an
actual charge of **$0.08**. The provider's charge is the experiment cost;
the longer helper interval also includes provisioning, readiness, retrieval,
and termination polling.

## Environment and test

- Lambda image: `lambda-stack-24-04`
- GPU: NVIDIA A10, 23,028 MiB, compute capability 8.6
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

Speedup is the median integrated CUDA32 time divided by median integrated
Tensor-path time. Above 1.0 means the Tensor path was faster.

The exact `validated` hybrid only crossed CUDA32 for coherent primary rays at
large leaves: 1.075× at leaf 128 and 1.044× at leaf 256. It did not beat
CUDA32 for shuffled primary rays or any measured secondary-ray case. This is
the main result: with useful BVH selectivity, ordinary CUDA32
Möller–Trumbore remained faster.

The no-Möller approximate paths showed larger dense-leaf crossovers:

| Workload | Best `uvt-depthsorted` | Best `e0e1e2` |
|---|---:|---:|
| Primary, coherent | 1.638× at leaf 128 | 1.628× at leaf 128 |
| Primary, shuffled | 0.955× at leaf 128 | 0.964× at leaf 128 |
| Secondary, pixel-ordered | 1.307× at leaf 64 | 1.305× at leaf 64 |
| Secondary, shuffled | 1.229× at leaf 128 | 1.223× at leaf 128 |

These are same-tree dense-candidate comparisons, not claims against an
optimally selective renderer or hardware RT cores.

## Accuracy result

The validated path matched the reference in every archived comparison. The
Tensor-owned variants intentionally make the FP16-input Tensor result own hit
inclusion and depth, so their disagreement is part of the evidence:

| Variant and ray set | Max false positives/rays | Max false negatives/reference hits | Max wrong primitive/reference hits | Max relative depth error |
|---|---:|---:|---:|---:|
| `uvt-depthsorted`, primary | 0.0027% | 0.2452% | 0.2758% | 7.73% |
| `e0e1e2`, primary | 0.0027% | 0.1839% | 0.3065% | 21.1% |
| `uvt-depthsorted`, secondary | 0.0000% | 0.4710% | 0.4710% | 242% |
| `e0e1e2`, secondary | 0.0000% | 0.1570% | 0.7849% | 3.44% |

The 242% entry occurred in a secondary case that also reported wrong
primitives; the aggregate counter does not establish that the same ray caused
both. That case's maximum absolute depth difference was 0.113 scene units; the
archive-wide maximum absolute difference was 0.547 in a primary `e0e1e2`
case. These are worst cases, not typical depth errors.

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
- `bin/`: the two SM86 executables built on the rental; rebuild from source
  instead of treating them as portable binaries.

The procedural Grid requires no third-party geometry or textures. This bundle
therefore contains no external scene assets.

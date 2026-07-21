# Provisional RTX 3050 Ti Grid matched-packet spot check

This is an internally intact raw-sample bundle from a pre-public RayMMA
development tree. It uses only the procedural, redistributable Grid scene. It
is not current release evidence and not a general GPU-performance claim.

The comparison is the `validated` WMMA-filter-plus-FP32-Möller hybrid against
the matched 16-ray CUDA diagnostic. It contains no independent-ray CUDA32
timing, selective-leaf result, secondary rays, current row normalization, or
`uvt-depthsorted`/`e0e1e2` result.

## Source and command

- Source identity: pre-public development tree, identified by the source hash
  below and retained only as historical context.
- `src/research_benchmark.cu` SHA-256:
  `d2abd50e9d02346d54a97984e24cd882f27c56a57b10f327740a7405d90f6b22`
- Configuration: Release, CUDA architecture 86

```sh
cmake --preset core -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset core --parallel
./build/core/tensor-wide-bvh-bench \
  --candidate-rich --scene Grid \
  --raw-csv raw.csv
```

## Provisional result

Median integrated trace-kernel time, 256x144, nine samples:

| Ray order | Matched CUDA-packet16 diagnostic | Validated hybrid | packet16 / hybrid |
|---|---:|---:|---:|
| coherent | 1.3763 ms | 1.4326 ms | 0.961x |
| packet-shuffled | 2.5702 ms | 2.4546 ms | 1.047x |

Both orders passed the checks implemented by that development revision:
full-image Tensor/CUDA comparison, strict 256-ray brute-force sampling,
primitive agreement, depth tolerance, numerical fallback reporting, and
packet-leaf overflow checks.

The result is deliberately modest: the validated hybrid lost for coherent
rays and reported a `1.047x` ratio for packet-shuffled primary rays. The
separated Tensor leaf kernel was slower in both orders. This does not establish
an advantage over CUDA32, production rays, or production tracing systems.

## Files

- `raw.csv`: every CUDA-event sample for all five timing scopes.
- `stdout.txt`: unmodified benchmark console output.
- `environment.txt`: sanitized system, GPU, toolchain, and build metadata.

SHA-256:

```text
768fe63671232c113119a940fdc4e5534bab21442f8445755cf0e9f84d1fed83  raw.csv
09457e7f2aa86ef5dff4d00f980c010a9e382843d6a515c04e8c3d2583e7c207  stdout.txt
b7c6e8259bd34afdf61075376e1af2451c8b24c6e22f1e0f5287ba9c09e21ad2  environment.txt
```

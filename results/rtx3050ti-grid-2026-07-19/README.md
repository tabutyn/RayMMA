# RTX 3050 Ti Grid spot check

This is the first raw-sample bundle produced by the hardened RayMMA harness.
It uses only the procedural, redistributable Grid scene. It is a single-run
release-candidate spot check, not a general GPU-performance claim.

## Source and command

- Source identity: pre-public development tree; rerun from the final public
  release commit before treating this as archival publication evidence.
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

## Result

Median integrated trace-kernel time, 256x144, nine samples:

| Ray order | Matched CUDA | Tensor | Speedup |
|---|---:|---:|---:|
| coherent | 1.3763 ms | 1.4326 ms | 0.961x |
| packet-shuffled | 2.5702 ms | 2.4546 ms | 1.047x |

Both orders passed full-image Tensor/CUDA comparison, strict 256-ray
brute-force sampling, primitive agreement, depth tolerance, numerical fallback
reporting, and packet-leaf overflow checks.

The result is deliberately modest: Tensor lost for coherent rays and won by
4.7% for packet-shuffled primary rays. The separated Tensor leaf kernel was
slower in both orders. This does not establish an advantage for production
rays or production tracing systems.

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

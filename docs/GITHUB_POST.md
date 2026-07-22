# LinkedIn post

Suggested image: [`cloud-gpu-crossover.png`](assets/cloud-gpu-crossover.png)

## Post

When do Tensor Cores actually help ray tracing?

I built RayMMA and ran the same CUDA benchmark on NVIDIA A10, A100, and H100
GPUs through Lambda Cloud.

A ray tracer uses a spatial index called a BVH to discard most triangles. A
“leaf” is the small batch left to test. The chart compares an FP16 Tensor Core
path with ordinary CUDA triangle intersection on the same BVH. Above 1.0x,
the Tensor path is faster.

Through a maximum of 32 triangles per leaf, the Tensor median stayed at or
below parity; A100 at 32 was nearly tied. At a maximum of 64, the Tensor path
crossed over on all three. At 128 it reached:

- A10: 1.64x
- A100: 1.72x
- H100: 1.59x

The important caveat: the plotted fast path is approximate. It lets the Tensor
result decide the hit and depth. Across both approximate modes and all cloud
ray sets, the worst cases reached 0.471% missed reference hits and 0.785% wrong
closest triangles. One secondary-ray case also reached 242% maximum relative
depth error. The exact validated hybrid kept only a small gain on A10/A100 and
none on H100. A selective BVH with ordinary CUDA remained the fastest
complete configuration.

All three rentals built from public commits, passed all 16 tests, saved the
raw timing and correctness data, and were automatically terminated. The actual
Lambda bill was $0.32 total: $0.08 for A10, $0.05 for A100, and $0.19 for
H100. The helper combined Lambda's API for lifecycle operations with SSH for
the build and verified result download.

I also watched for a single B200 139 times over 12 hours overnight. None
became available; B200 was not benchmarked, and no B200 charge was incurred.

The useful lesson was not “Tensor Cores make ray tracing faster.” It was that
dense batches can make matrix hardware worthwhile, while good filtering and
accuracy often matter more than raw arithmetic throughput.

Code, graph, scripts, and raw evidence:
https://github.com/tabutyn/RayMMA

#CUDA #GPUComputing #RayTracing #PerformanceEngineering

## Image alt text

Two-panel RayMMA chart. The left panel plots approximate Tensor-path speedup
over CUDA32 against maximum BVH leaf size for A10, A100, and H100 GPUs. All
three are below parity through leaf 32, cross above parity at leaf 64, and
peak at leaf 128 between 1.59x and 1.72x. The right panel compares absolute
CUDA32 and Tensor latency at leaf 128; H100 is fastest for this workload. A
footer states that the plotted path is approximate, the exact hybrid has much
smaller gains, the total Lambda bill was $0.32, and no B200 capacity appeared
during 139 checks over 12 hours.

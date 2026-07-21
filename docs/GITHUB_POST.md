# LinkedIn post

Can Tensor Cores speed up ray tracing?

I built RayMMA to test that question. The answer turned out to be more useful
than a simple yes or no.

Tensor Cores are excellent at dense matrix math. Ray tracing usually does the
opposite: a spatial index called a BVH quickly removes irrelevant triangles,
leaving small and irregular batches of work.

RayMMA compares a Tensor Core ray/triangle method with the standard CUDA
Möller–Trumbore intersection test while keeping traversal and candidate
ordering consistent.

I tested it locally on an RTX 3050 Ti Laptop GPU, then paid for full archive
runs on Lambda Cloud NVIDIA A10, A100, and H100 GPUs. Each rental built from a
public commit, passed all 16 tests, retained every timing sample and correctness
counter, and were downloaded and checksum-verified before termination.

Across those experiments:

- With a selective BVH, CUDA Möller–Trumbore was clearly faster. The best
  configuration was about 9.3× faster for coherent rays and 3.3× faster for
  shuffled rays.
- When I deliberately made the candidate batches much larger, the approximate
  Tensor modes reached up to 1.64× on A10, 1.72× on A100, and 1.59× on H100.
- That gain was not free: a few hits and closest triangles were wrong, with up
  to 3.28% relative depth error in the measured run.

The broader lesson: making one computation faster does not necessarily make
the whole program faster. Data selection, batching, memory movement, and
accuracy can matter more than raw arithmetic throughput.

The repository includes the CUDA implementation, tests, raw benchmark data,
the complete paid A10, A100, and H100 evidence bundles, reproduction instructions, and
negative results.

https://github.com/tabutyn/RayMMA

#CUDA #GPUComputing #RayTracing #PerformanceEngineering

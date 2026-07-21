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

On an RTX 3050 Ti Laptop GPU:

- With a selective BVH, CUDA Möller–Trumbore was clearly faster. The best
  configuration was about 9.3× faster for coherent rays and 3.3× faster for
  shuffled rays.
- When I deliberately made the candidate batches much larger, the approximate
  Tensor modes reached 1.06–1.30× CUDA performance.
- That gain was not free: a few hits and closest triangles were wrong, with up
  to 3.28% relative depth error in the measured run.

The broader lesson: making one computation faster does not necessarily make
the whole program faster. Data selection, batching, memory movement, and
accuracy can matter more than raw arithmetic throughput.

The repository includes the CUDA implementation, tests, raw benchmark data,
reproduction instructions, and negative results.

https://github.com/tabutyn/RayMMA

#CUDA #GPUComputing #RayTracing #PerformanceEngineering

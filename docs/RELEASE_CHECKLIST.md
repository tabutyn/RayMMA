# Public release checklist

Use a new dedicated RayMMA repository. Do not make the complete
`ballroller-2026` repository public: it contains unrelated projects, private
submodule URLs, large history, and assets whose public redistribution has not
been cleared.

## Completed in this working tree

- [x] Scope MIT licensing to original RayMMA source and documentation.
- [x] Make the CUDA-only Grid benchmark the default build.
- [x] Remove the legacy comparison backend, source, build paths, and checkout.
- [x] Remove monorepo-relative inputs from the portable core build.
- [x] Rename `from_chat` to the public-facing `prototype`.
- [x] Add citation, contribution, security, changelog, CI, and method files.
- [x] Describe the result as a bounded candidate-density crossover.
- [x] Replace “incoherent” with “packet-shuffled primary rays.”
- [x] State that the FP16 envelope is empirical rather than formally proven.
- [x] Stratify brute-force samples 16-by-16 across original image coordinates
      and require strict primitive agreement for every correctness path.
- [x] Add odd leaf-layout, candidate-rich, and FP16 range regressions.
- [x] Add CSV export for every CUDA-event timing sample.
- [x] Archive a procedural Grid run with raw samples, console output,
      sanitized environment metadata, and checksums.
- [x] Validate a curated copy without Ball Roller directories or third-party
      SDK headers: standalone Release build and all nine core tests pass.

## Must be completed before the repository becomes public

- [ ] Confirm that `Thomas Butyn` is the correct copyright holder and that no
      employer, client, school, or collaborator agreement owns the work.
- [ ] If patent protection might matter, obtain IP advice before the first
      public disclosure. Public release can destroy rights in some countries.
- [ ] Create a clean repository containing only the curated RayMMA files.
      Exclude Ball Roller history and all Checker GLBs/textures.
- [x] Omit the inspected MIT ray-tracer checkout and preserve its complete
      upstream MIT notice in `THIRD_PARTY.md`.
- [x] Keep TinyBVH optional, external, pinned, and license-documented.
- [ ] Run a dedicated secret scanner on the curated repository and inspect the
      complete new history.
- [x] Build and test the exact allowlisted export with no Ball Roller
      directories present.
- [ ] Run CPU CI and a documented self-hosted GPU job.
- [ ] Rerun the benchmark from the release commit and save raw samples,
      environment metadata, commands, hashes, and correctness output.
- [x] Add genuine first-bounce diffuse secondary rays.
- [x] Add an independent-ray 32-ray/warp CUDA control.
- [x] Add established TinyBVH SAH and spatial-split builder controls.
- [ ] Replace or clear the ownership of Checker assets before showing them in a
      downloadable release or source history.

## GitHub release

- [ ] Use the repository description:
      `CUDA/WMMA research artifact for ray-triangle candidate filtering, with tuned CUDA and production-BVH controls.`
- [ ] Add topics such as `cuda`, `gpu`, `ray-tracing`, `tensor-cores`, `wmma`,
      `bvh`, and `computer-graphics`.
- [ ] Add one architecture diagram, one crossover plot, and a short captured
      demo that contains no uncleared assets.
- [ ] Set `repository-code` in `CITATION.cff` after the final URL exists.
- [ ] Tag `v0.1.0`, publish checksums, and attach raw result artifacts.
- [ ] Optionally archive the tag with Zenodo, then add its DOI to
      `CITATION.cff`.

Create an inspectable allowlisted tree with:

```sh
./tools/export_public_tree.sh /path/to/new/raymma
```

The helper deliberately does not run `git init`, commit, or publish anything.
Inspect it, run a secret scan, and repeat the standalone build before creating
the public repository.

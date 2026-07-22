# Results policy

Publish raw measurements, not hand-copied medians alone.

The retained evidence bundles are:

- the [paid Lambda Cloud A100 archive run](lambda-a100-2026-07-21/README.md),
  built natively for SM80 with the complete primary and secondary suite;
- the [paid Lambda Cloud A10 archive run](lambda-a10-2026-07-21/README.md),
  with all 16 tests, primary and secondary procedural Grid leaf sweeps, raw
  nine-sample CSVs, complete transcripts, machine/source provenance, built
  executables, and the original verified download;
- the [paid Lambda Cloud H100 archive run](lambda-h100-2026-07-21/README.md),
  with the same archive contract built natively for SM90;
- the [RTX 3050 Ti Coastal Cliff rerun](rtx3050ti-coastal-cliff-2026-07-21/README.md),
  with the tested exact and Tensor-owned backends, 15 passing tests, raw
  nine-sample CSVs, full transcripts, environment capture, source and asset
  hashes, and archive checksums; and
- the [12-hour Lambda B200 availability record](lambda-b200-availability-2026-07-21/README.md),
  which records 139 successful checks but is explicitly not a performance
  benchmark because no B200 instance became available or launched.

[`cloud-gpu-comparison-2026-07-21.csv`](cloud-gpu-comparison-2026-07-21.csv)
is a reproducible median-only view derived from the original A10/A100/H100
sample CSVs for the [published comparison graph](../docs/assets/cloud-gpu-crossover.svg).
The original per-sample files remain authoritative.

The retained bundles follow this evidence contract:

- `environment.txt` from `tools/capture_environment.sh`;
- exact benchmark commands and GPU preconditioning procedure;
- unmodified stdout/stderr and all samples from `--raw-csv FILE`;
- scene source, license, SHA-256, and triangle count;
- tested Git commit and dirty status; and
- every hit, primitive, depth, overflow, and fallback counter.

The archived no-Möller claims identify the Tensor variant and report its
accuracy counters beside its speedup.

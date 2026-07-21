# Results policy

Publish raw measurements, not hand-copied medians alone.

The current evidence bundle is the
[RTX 3050 Ti Coastal Cliff rerun](rtx3050ti-coastal-cliff-2026-07-21/README.md).
It contains the current exact and Tensor-owned backends, 15 passing tests,
nine-sample timing CSVs, full transcripts, environment capture, source and
asset hashes, and archive checksums.

For a new machine or workload, retain:

- `environment.txt` from `tools/capture_environment.sh`;
- exact benchmark commands and GPU preconditioning procedure;
- unmodified stdout/stderr and all samples from `--raw-csv FILE`;
- scene source, license, SHA-256, and triangle count;
- tested Git commit and dirty status; and
- every hit, primitive, depth, overflow, and fallback counter.

No-Möller claims must identify the Tensor variant and report its accuracy
counters beside its speedup.

For an A100, `tools/run_a100.sh --profile archive` creates a ready-to-transfer
tarball containing the procedural Grid primary/secondary leaf sweeps, all
three variants, the build and test logs, environment, binaries, raw CSVs, and
checksums. Keep the `.sha256` sidecar with the tarball.

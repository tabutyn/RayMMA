# Results policy

Do not commit hand-copied medians as if they were raw evidence.

The public interpretation and evidence-status matrix live in
[`docs/RESULTS.md`](../docs/RESULTS.md). A directory here is evidence only for
the exact source, backend, baseline, scene, and workload it records.

For each machine, create a directory containing:

- `environment.txt` from `tools/capture_environment.sh`;
- exact benchmark commands;
- unmodified stdout/stderr;
- raw per-launch samples from `--raw-csv FILE`;
- scene source, license, and SHA-256;
- the tested Git commit and dirty status; and
- a short README defining every timing scope and speedup.

Historical RTX 3050 Ti medians are documented in
[`docs/RESULTS_RTX3050TI.md`](../docs/RESULTS_RTX3050TI.md). Their raw samples
were not retained, so they are hypothesis-generating notes rather than
publication evidence.

The retained
[`2026-07-19 RTX 3050 Ti Grid spot check`](rtx3050ti-grid-2026-07-19/README.md)
contains raw samples and internally verified checksums, but it is provisional:
its pre-public source commit is not reconstructable from the clean repository,
and it predates CUDA32 and the no-Möller variants.

The
[`2026-07-21 public-commit archival rerun`](rtx3050ti-pushed-5e05391-2026-07-21/README.md)
contains CUDA32, leaf-16 and leaf-256 scene suites, a Checker High leaf sweep,
secondary rays, TinyBVH builder controls, complete raw samples, input hashes,
secret/license audit reports, and checksums. It is version-bounded evidence for
the older validated-only commit `5e05391`; that commit does not contain the
no-Möller variants or current normalization.

The
[`2026-07-21 Coastal Cliff rerun`](rtx3050ti-coastal-cliff-2026-07-21/README.md)
contains the current validated and no-Möller backends, 15 passing tests, all
raw timing samples, complete transcripts, environment capture, and checksums.
It uses the CC0 replacement model and records exact miss, primitive, and depth
errors. Its source checksum manifest identifies the exact tested implementation
independently of later documentation edits.

No-Möller claims must include the exact Tensor variant, CSV schema, and every
accuracy counter.

# Results policy

Do not commit hand-copied medians as if they were raw evidence.

For each machine, create a directory containing:

- `environment.txt` from `tools/capture_environment.sh`;
- exact benchmark commands;
- unmodified stdout/stderr;
- raw per-launch samples from `--raw-csv FILE`;
- scene source, license, and SHA-256;
- the tested Git commit and dirty status; and
- a short README defining every timing scope and speedup.

The historical RTX 3050 Ti medians are documented in
[`docs/RESULTS_RTX3050TI.md`](../docs/RESULTS_RTX3050TI.md). Their raw samples
were not retained, so they are not represented here as publication evidence.

The first auditable bundle is the
[`2026-07-19 RTX 3050 Ti Grid spot check`](rtx3050ti-grid-2026-07-19/README.md).

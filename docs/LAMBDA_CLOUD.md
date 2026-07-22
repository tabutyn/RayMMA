# Running RayMMA on Lambda Cloud

`tools/lambda_cloud.py` automates the useful lifecycle through Lambda's
official REST API:

1. inspect live single-GPU B200 capacity and prices;
2. upload a dedicated SSH **public** key if needed;
3. create or reuse a region-specific TCP/22 ruleset for your source CIDR;
4. launch one supported GPU with no persistent filesystem;
5. wait for the instance and SSH;
6. install only missing build packages, clone the public RayMMA ref, and run
   `tools/run_cloud_gpu.sh`;
7. download and verify the result tarball, with bounded retries after an
   interrupted SSH run; and
8. terminate through the API and poll for confirmation, including after most
   failures or `Ctrl-C`.

At the time of the July 2026 runs, Lambda published a REST API and an OpenAPI
specification, but not a dedicated official CLI or language SDK. The Python
helper uses only the standard library plus local OpenSSH commands.

## One unavoidable account step

Lambda's documented API does not create API keys. Finish account, billing,
and terms setup, then use the console's **API keys → Generate API Key** once.
After that, the benchmark lifecycle needs no website.

Load the key into only the current shell; do not put it in this repository or
on the rental:

```sh
read -rsp 'Lambda API key: ' LAMBDA_API_KEY
export LAMBDA_API_KEY
printf '\n'
python3 tools/lambda_cloud.py inventory
```

The API key has broad account authority. Unset it when finished:

```sh
unset LAMBDA_API_KEY
```

The helper pins that Bearer credential to `https://cloud.lambda.ai/api/v1`.
`inventory --json` redacts Jupyter tokens and token-bearing Jupyter URLs
before printing API records.

## Create a rental-only SSH key

Create this locally. The private key stays on your computer; the helper sends
only the `.pub` file to Lambda:

```sh
ssh-keygen -t ed25519 -N '' \
  -f "$HOME/.ssh/raymma_lambda" -C raymma-lambda
chmod 600 "$HOME/.ssh/raymma_lambda"
```

Find your current public IPv4 address using a source you trust. For example:

```sh
MY_IP="$(curl -4fsS https://icanhazip.com)"
printf 'SSH source: %s/32\n' "$MY_IP"
```

Use a fixed office/VPN CIDR instead if your address changes during a run.

## Preview, then run

Commit and push the exact RayMMA revision before renting the GPU. The helper
clones the public `--ref`; it cannot see uncommitted or merely local changes.
Confirm the remote ref without opening GitHub:

```sh
git status --short
git ls-remote --exit-code origin refs/heads/main
```

The dry run performs read-only API discovery and prints the exact launch plan.
It does not upload the key, create a firewall ruleset, launch an instance, or
create an output directory:

```sh
python3 tools/lambda_cloud.py run \
  --ssh-public-key "$HOME/.ssh/raymma_lambda.pub" \
  --ssh-private-key "$HOME/.ssh/raymma_lambda" \
  --ssh-cidr "$MY_IP/32" \
  --profile quick \
  --dry-run
```

Run the short path interactively by omitting `--yes`; type `launch` when the
price and region look right. For a non-interactive archival run:

```sh
python3 tools/lambda_cloud.py run \
  --ssh-public-key "$HOME/.ssh/raymma_lambda.pub" \
  --ssh-private-key "$HOME/.ssh/raymma_lambda" \
  --ssh-cidr "$MY_IP/32" \
  --profile archive \
  --output-dir "$PWD/lambda-results" \
  --yes
```

For a bounded unattended B200 availability watch, keep the credential in a
permission-0600 file outside the repository and run:

```sh
LAMBDA_API_KEY_FILE="/secure/path/lambda-api-key" \
  ./tools/watch_lambda_b200.sh --hours 12
```

Replace `/secure/path/lambda-api-key` with the local path to the protected key
file.

The watcher uses one local lock, randomized three-to-seven-minute intervals,
the B200 type only, and a hard $7.00/hour live-price ceiling. If capacity
appears within that cap, it runs the same archive workflow once and exits after
verified retrieval and confirmed termination. Progress is written to
`lambda-results/b200-watch.state`; redirect stdout/stderr to a log when
launching it detached. The watcher cannot survive the local computer powering
off or losing network access.

The retained July 21–22 attempt completed a 12-hour overnight wall-clock
window and 139 successful API checks. No eligible single B200 appeared, so it
launched nothing and incurred no B200 compute charge. See the
[availability record](../results/lambda-b200-availability-2026-07-21/README.md).

Selection is based on the live `/instance-types` response. The helper accepts
only the B200 family with a reported GPU count of exactly one; price and type
name break ties. Use `--instance-type` and `--region` to pin a choice shown by
`inventory`. It will not silently rent a multi-GPU machine or a GPU family
outside that list.

The default base image is `lambda-stack-24-04`: it supplies the NVIDIA driver
and CUDA toolkit, while Ubuntu 24.04 supplies a new enough CMake for RayMMA.
The bootstrap runs `apt-get update` and installs required build essentials,
CMake, Git, Python, and CA certificates. It never runs `upgrade`,
`full-upgrade`, or `dist-upgrade`. Use `--skip-bootstrap` only after verifying
the image tools.

The public checkout defaults to `https://github.com/tabutyn/RayMMA.git` at
`main`. `--ref` accepts a public branch, tag, or commit. No GitHub credential
is copied to the instance.

The rental builds from source instead of trusting a locally cross-compiled
binary. The `core` preset selects the rental GPU's native CUDA architecture.
CUDA executables still depend on driver/toolkit compatibility, while this
project is small enough that native compilation is cheap. The exact binaries
are included in the downloaded evidence archive for later inspection.

## Results on your computer

Each run creates `lambda-results/<instance-name>/` containing:

- `launch-plan.json` and the instance ID;
- a dedicated `known_hosts` file;
- the remote console transcript;
- `raymma-cloud-results.tar.gz`; and
- `raymma-cloud-results.tar.gz.sha256`.

On a successful transfer, the helper verifies SHA-256 locally before
termination. Inside the archive are the commit and source hashes, environment,
CMake and CTest logs, raw CSV samples, full benchmark transcripts, exit code,
and native-architecture executables. The `archive` profile covers primary and
deterministic secondary rays for `validated`, `uvt-depthsorted`, and `e0e1e2`
over the procedural Grid leaf sweep. It downloads no geometry or texture
assets.

## Firewall behavior

The helper never edits the workspace-global firewall. It attaches a
per-instance ruleset allowing only TCP/22 from `--ssh-cidr`; no benchmark port
is opened. The ruleset and uploaded public key are reusable and do not incur
compute charges.

Lambda documents global rules as applying to all instances, but does not say
that an attached ruleset overrides them. The helper therefore fails closed if
the global policy permits SSH from outside `--ssh-cidr`. Resolve that
workspace-wide policy separately if strict source restriction is required;
changing it inside a benchmark script could disconnect unrelated instances.
`--allow-broad-global-ssh` is an explicit opt-out that keeps Lambda's SSH-key
authentication but accepts the wider global source rule.

If a new workspace still has Lambda's default world-source SSH rule, either
narrow the global ruleset through the documented API first or add
`--allow-broad-global-ssh` after consciously accepting public port 22 with
key-only authentication. The helper will not make that account-wide choice
implicitly.

Lambda documents that firewall rules do not apply in `us-south-1`, so the
helper excludes that region from automatic and explicit GPU selection.

## Retained paid runs

The July 21, 2026 [A10](../results/lambda-a10-2026-07-21/README.md),
[A100](../results/lambda-a100-2026-07-21/README.md), and
[H100](../results/lambda-h100-2026-07-21/README.md) bundles record complete
paid runs of this workflow. Every rental passed all 16 tests and the archive
suite, downloaded and verified its result, confirmed termination, and was
followed by an inventory query showing no running instances.

The provider's billing history reports actual charges of $0.08 for A10,
$0.05 for A100, and $0.19 for H100: **$0.32 total**. The helper printed larger
conservative estimates because it applied each hourly rate to the entire
launch-request-through-termination-confirmation wall-clock interval, including
provisioning, readiness, retrieval, and shutdown polling. Provider billing is
authoritative. See [Findings and evidence](RESULTS.md) for the performance
comparison and reproducible graph.

## Failure and billing safety

By default, `finally` makes bounded best-effort retrieval attempts, sends the
termination request, and polls until the instance is terminated or absent.
Retrieval failure never prevents termination. Do not use `shutdown` or
`poweroff`: Lambda documents that billing continues and the instance can enter
an Alert state. `--keep-instance` is an explicit opt-out and prints the
still-billing instance ID.

No local program can run cleanup after `kill -9`, a laptop power loss, or a
lost launch response. Before launch, the helper saves a random UUID run tag
with the expected name, type, and region. It never retries launch blindly and
will only adopt an instance matching that full recovery plan. Recovery after
a local process or machine loss is entirely command-line:

```sh
python3 tools/lambda_cloud.py inventory
python3 tools/lambda_cloud.py terminate --instance-id INSTANCE_ID
```

The local `recovery-plan.json` records the random `run_id`; `inventory --json`
can be searched for that exact tag without exposing Jupyter credentials.

The script prints Lambda's live hourly price and a minute-rounded estimate;
the provider invoice remains authoritative. Capacity is first-come and can
disappear between inventory and launch.

## Official references

- [Lambda Cloud API browser and OpenAPI specification](https://docs-api.lambda.ai/api/cloud)
- [OpenAPI 1.10.0 JSON](https://docs-api.lambda.ai/api/cloud/spec.json)
- [On-Demand instance types and base images](https://docs.lambda.ai/public-cloud/on-demand/)
- [Creating, polling, and terminating instances](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/)
- [SSH connection guidance](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)
- [Firewall defaults and rulesets](https://docs.lambda.ai/public-cloud/firewalls/)
- [Importing and exporting data](https://docs.lambda.ai/public-cloud/importing-exporting-data/)
- [Billing](https://docs.lambda.ai/public-cloud/billing/)

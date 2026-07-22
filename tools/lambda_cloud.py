#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch, benchmark, retrieve, and terminate a Lambda Cloud GPU."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from typing import Any
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_API = "https://cloud.lambda.ai/api/v1"
DEFAULT_REPOSITORY = "https://github.com/tabutyn/RayMMA.git"
DEFAULT_IMAGE_FAMILY = "lambda-stack-24-04"
POLL_SECONDS = 5.0
FIREWALL_UNSUPPORTED_REGIONS = {"us-south-1"}


class CloudError(RuntimeError):
    """A Lambda API, SSH, build, or retrieval failure."""


class AmbiguousLaunchError(CloudError):
    """The launch request outcome is unknown and must not be retried blindly."""


class APIHTTPError(CloudError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"Lambda API {code}: {message}")
        self.status = status
        self.code = code


class APITransportError(CloudError):
    """No authoritative HTTP response was received."""


class MultipleLaunchError(CloudError):
    def __init__(self, instance_ids: list[str]) -> None:
        super().__init__(f"Launch unexpectedly returned multiple IDs: {instance_ids}")
        self.instance_ids = instance_ids


class DefinitiveLaunchError(CloudError):
    """The API authoritatively rejected launch before creating an instance."""


class LambdaAPI:
    def __init__(self, api_key: str, base_url: str = DEFAULT_API) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise CloudError("The API base must use HTTPS (except localhost tests).")
        self.last_request = 0.0

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        delay = 1.05 - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Lambda's Cloudflare edge rejects urllib's default
            # ``Python-urllib/*`` signature before the API sees the request.
            "User-Agent": "RayMMA-Lambda-Runner/1.0 (+https://github.com/tabutyn/RayMMA)",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            self.last_request = time.monotonic()
            with urlopen(request, timeout=35) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                details = json.loads(raw).get("error", {})
                message = details.get("message", raw)
                code = details.get("code", f"HTTP {error.code}")
            except json.JSONDecodeError:
                code, message = f"HTTP {error.code}", raw
            raise APIHTTPError(error.code, str(code), str(message)) from error
        except (URLError, TimeoutError, OSError) as error:
            raise APITransportError(f"Lambda API transport error: {error}") from error
        except json.JSONDecodeError as error:
            raise CloudError("Lambda API returned invalid JSON.") from error
        if not isinstance(payload, dict) or "data" not in payload:
            raise CloudError("Lambda API returned an unexpected response shape.")
        return payload["data"]


def api_from_args(args: argparse.Namespace) -> LambdaAPI:
    api_key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not api_key:
        raise CloudError(
            "LAMBDA_API_KEY is unset. Generate it once in the Lambda console, "
            "then export it only in your current shell."
        )
    return LambdaAPI(api_key, DEFAULT_API)


def region_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value or "")


GPU_PRIORITY = ("b200",)


def available_cloud_gpus(instance_types: dict[str, Any]) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    for key, record in instance_types.items():
        details = record.get("instance_type", {})
        name = str(details.get("name") or key)
        text = " ".join(
            str(details.get(field, ""))
            for field in ("name", "description", "gpu_description")
        ).lower()
        tokens = set(re.sub(r"[^a-z0-9]+", " ", text).split())
        specs = details.get("specs", {})
        try:
            gpu_count = int(specs.get("gpus", 0))
        except (TypeError, ValueError):
            gpu_count = 0
        regions = [
            region_name(region)
            for region in record.get("regions_with_capacity_available", [])
            if region_name(region)
            and region_name(region) not in FIREWALL_UNSUPPORTED_REGIONS
        ]
        model = next((item for item in GPU_PRIORITY if item in tokens), None)
        if model is None or gpu_count != 1 or not regions:
            continue
        choices.append(
            {
                "name": name,
                "description": str(details.get("description", "")),
                "gpu": str(details.get("gpu_description", model.upper())),
                "model": model,
                "price_cents_per_hour": int(
                    details.get("price_cents_per_hour", 0) or 0
                ),
                "regions": sorted(regions),
            }
        )
    return sorted(
        choices,
        key=lambda item: (
            GPU_PRIORITY.index(item["model"]),
            item["price_cents_per_hour"],
            item["name"],
        ),
    )
def choose_capacity(
    instance_types: dict[str, Any], requested_type: str | None, requested_region: str | None
) -> dict[str, Any]:
    choices = available_cloud_gpus(instance_types)
    if requested_type:
        choices = [item for item in choices if item["name"] == requested_type]
        if not choices:
            raise CloudError(
                f"{requested_type!r} is not an available supported single-GPU type. "
                "Run the inventory command to inspect live capacity."
            )
    if requested_region:
        choices = [item for item in choices if requested_region in item["regions"]]
    if not choices:
        suffix = f" in {requested_region}" if requested_region else ""
        raise CloudError(f"No supported single-GPU capacity is currently available{suffix}.")
    choice = choices[0].copy()
    choice["region"] = requested_region or choice["regions"][0]
    return choice


def enforce_price_cap(choice: dict[str, Any], maximum_cents: int | None) -> None:
    if maximum_cents is None:
        return
    if maximum_cents <= 0:
        raise CloudError("--max-price-cents must be a positive integer.")
    actual = int(choice["price_cents_per_hour"])
    if actual > maximum_cents:
        raise CloudError(
            f"Live price ${actual / 100:.2f}/h exceeds the unattended cap "
            f"${maximum_cents / 100:.2f}/h."
        )


def validate_image(images: list[dict[str, Any]], family: str, region: str) -> None:
    matching = [
        image
        for image in images
        if image.get("family") == family
        and region_name(image.get("region")) == region
        and image.get("architecture") == "x86_64"
    ]
    if not matching:
        raise CloudError(f"Image family {family!r} is not available in {region} for x86_64.")


def normalize_public_key(value: str) -> str:
    parts = value.strip().split()
    key_type = parts[0] if parts else ""
    if len(parts) < 2 or not key_type.startswith(("ssh-", "ecdsa-", "sk-")):
        raise CloudError("The public key file does not contain an OpenSSH public key.")
    return " ".join(parts[:2])


def read_ssh_keys(public_path: Path, private_path: Path) -> tuple[str, Path, Path]:
    public_path = public_path.expanduser().resolve()
    private_path = private_path.expanduser().resolve()
    if not public_path.is_file():
        raise CloudError(f"SSH public key not found: {public_path}")
    if not private_path.is_file():
        raise CloudError(f"SSH private key not found: {private_path}")
    if private_path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CloudError(f"SSH private key permissions are too broad; run chmod 600 {private_path}")
    public_key = normalize_public_key(public_path.read_text())

    def fingerprint(path: Path) -> str:
        result = subprocess.run(
            ["ssh-keygen", "-lf", str(path), "-E", "sha256"],
            text=True,
            capture_output=True,
            check=False,
        )
        fields = result.stdout.split()
        if result.returncode or len(fields) < 2:
            raise CloudError(
                f"Could not read SSH key fingerprint for {path}: "
                f"{result.stderr.strip()}"
            )
        return fields[1]

    if fingerprint(public_path) != fingerprint(private_path):
        raise CloudError("The SSH public and private key files do not match.")
    return public_key, public_path, private_path


def ensure_ssh_key(
    api: LambdaAPI, name: str, public_key: str, dry_run: bool
) -> str:
    keys = api.request("GET", "ssh-keys")
    matching = [key for key in keys if key.get("name") == name]
    if len(matching) > 1:
        raise CloudError(f"Multiple Lambda SSH keys are named {name!r}.")
    if matching:
        remote = matching[0].get("public_key")
        if not remote:
            raise CloudError(
                f"Lambda SSH key {name!r} has no comparable public key; "
                "choose another --ssh-key-name."
            )
        if normalize_public_key(str(remote)) != public_key:
            raise CloudError(
                f"Lambda SSH key {name!r} exists with different key material; "
                "choose another --ssh-key-name."
            )
        print(f"Reusing Lambda SSH key: {name}")
        return str(matching[0].get("id", "existing"))
    if dry_run:
        print(f"DRY RUN: would upload SSH public key {name!r}")
        return "dry-run-ssh-key"
    created = api.request(
        "POST", "ssh-keys", {"name": name, "public_key": public_key}
    )
    print(f"Uploaded Lambda SSH public key: {name}")
    return str(created.get("id", "created"))


def expected_firewall_rule(cidr: str) -> dict[str, Any]:
    return {
        "protocol": "tcp",
        "port_range": [22, 22],
        "source_network": cidr,
        "description": "RayMMA SSH",
    }


def firewall_rule_key(rule: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rule.get("protocol"),
        tuple(rule.get("port_range") or []),
        str(rule.get("source_network", "")),
    )


def check_global_firewall(
    api: LambdaAPI, allowed_cidr: str, allow_additional_ssh: bool
) -> None:
    try:
        global_rules = api.request("GET", "firewall-rulesets/global").get("rules", [])
    except CloudError as error:
        if allow_additional_ssh:
            print(f"Warning: could not inspect global firewall rules: {error}", file=sys.stderr)
            return
        raise CloudError(
            f"Could not inspect the workspace-global firewall: {error}. "
            "Refusing a billable launch because SSH exposure is unknown."
        ) from error
    intended = ipaddress.ip_network(allowed_cidr)
    additional = []
    for rule in global_rules:
        ports = rule.get("port_range") or []
        protocol = rule.get("protocol")
        source = rule.get("source_network")
        covers_ssh = protocol in {"all", "tcp"} and (
            protocol == "all" or (len(ports) == 2 and ports[0] <= 22 <= ports[1])
        )
        if not covers_ssh:
            continue
        try:
            source_network = ipaddress.ip_network(str(source), strict=False)
        except ValueError:
            additional.append(rule)
            continue
        if source_network.version != 4 or not source_network.subnet_of(intended):
            additional.append(rule)
    if not additional:
        return
    message = (
        "The workspace-global firewall allows SSH outside the requested "
        f"{allowed_cidr}. Lambda does not document an attached ruleset as an "
        "override, so the /CIDR rule cannot be claimed as restrictive."
    )
    if not allow_additional_ssh:
        raise CloudError(
            message
            + " Narrow the global rule first, or explicitly accept it with "
            "--allow-broad-global-ssh."
        )
    print(f"Warning: {message}", file=sys.stderr)


def ensure_firewall_ruleset(
    api: LambdaAPI, name: str, region: str, cidr: str, dry_run: bool
) -> str:
    rulesets = api.request("GET", "firewall-rulesets")
    matching = [
        ruleset
        for ruleset in rulesets
        if ruleset.get("name") == name
        and region_name(ruleset.get("region")) == region
    ]
    expected = expected_firewall_rule(cidr)
    if len(matching) > 1:
        raise CloudError(f"Multiple firewall rulesets are named {name!r} in {region}.")
    if matching:
        actual = matching[0].get("rules", [])
        if len(actual) != 1 or firewall_rule_key(actual[0]) != firewall_rule_key(expected):
            raise CloudError(
                f"Firewall ruleset {name!r} exists with different rules; "
                "choose another --firewall-name."
            )
        print(f"Reusing per-instance firewall ruleset: {name}")
        return str(matching[0]["id"])
    if dry_run:
        print(f"DRY RUN: would create firewall ruleset {name!r} in {region}")
        return "dry-run-firewall"
    created = api.request(
        "POST",
        "firewall-rulesets",
        {"name": name, "region": region, "rules": [expected]},
    )
    print(f"Created per-instance firewall ruleset: {name}")
    return str(created["id"])


def ssh_options(private_key: Path, known_hosts: Path) -> list[str]:
    return [
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def tag_value(instance: dict[str, Any], key: str) -> str | None:
    for tag in instance.get("tags") or []:
        if tag.get("key") == key:
            return str(tag.get("value", ""))
    return None


def validate_instance_id(instance_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]{8,128}", instance_id):
        raise CloudError(f"Lambda returned an invalid instance ID: {instance_id!r}")
    return instance_id


def instance_matches_plan(instance: dict[str, Any], plan: dict[str, str]) -> bool:
    instance_type = instance.get("instance_type") or {}
    actual_type = region_name(instance_type)
    actual_region = region_name(instance.get("region"))
    return (
        tag_value(instance, "run-id") == plan["run_id"]
        and tag_value(instance, "project") == "raymma"
        and instance.get("name") == plan["name"]
        and actual_type == plan["instance_type"]
        and actual_region == plan["region"]
    )


def find_instance_by_plan(
    api: LambdaAPI, plan: dict[str, str]
) -> dict[str, Any] | None:
    tagged = [
        item
        for item in api.request("GET", "instances")
        if tag_value(item, "run-id") == plan["run_id"]
    ]
    matches = [item for item in tagged if instance_matches_plan(item, plan)]
    if tagged and not matches:
        raise AmbiguousLaunchError(
            "An instance has this recovery tag but does not match the planned "
            "name, type, and region; refusing to adopt or terminate it."
        )
    if len(matches) > 1:
        raise AmbiguousLaunchError(
            "Multiple instances have the supposedly unique recovery tag; inspect "
            "them with inventory."
        )
    return matches[0] if matches else None


def recover_launch(
    api: LambdaAPI, plan: dict[str, str], attempts: int = 12
) -> dict[str, Any] | None:
    for _ in range(attempts):
        found = find_instance_by_plan(api, plan)
        if found:
            return found
        time.sleep(POLL_SECONDS)
    return None


def launch_instance(
    api: LambdaAPI, body: dict[str, Any], recovery_plan: dict[str, str]
) -> str:
    try:
        response = api.request("POST", "instance-operations/launch", body)
    except APIHTTPError as error:
        if error.status < 500:
            raise DefinitiveLaunchError(str(error)) from error
        ambiguous_error: CloudError = error
    except (APITransportError, CloudError) as error:
        ambiguous_error = error
    else:
        instance_ids = [str(value) for value in response.get("instance_ids", [])]
        if len(instance_ids) == 1:
            return validate_instance_id(instance_ids[0])
        if len(instance_ids) > 1:
            raise MultipleLaunchError(
                [validate_instance_id(value) for value in instance_ids]
            )
        ambiguous_error = AmbiguousLaunchError(
            "Launch returned success without an instance ID."
        )

    print(
        "Launch outcome is ambiguous. The endpoint is not idempotent; searching "
        "by the random recovery tag instead of retrying.",
        file=sys.stderr,
    )
    found = recover_launch(api, recovery_plan)
    if found:
        recovered_id = validate_instance_id(str(found.get("id", "")))
        print(f"Recovered launched instance by run tag: {recovered_id}")
        return recovered_id
    raise AmbiguousLaunchError(
        f"{ambiguous_error}\nNo matching instance appeared during recovery. "
        "Do not blindly rerun launch; use inventory and the saved run-id tag."
    ) from ambiguous_error


def wait_for_instance(api: LambdaAPI, instance_id: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        instance = api.request("GET", f"instances/{instance_id}")
        status_value = str(instance.get("status", "unknown"))
        address = str(instance.get("ip") or "")
        print(f"Instance {instance_id}: {status_value}{' ' + address if address else ''}")
        if status_value == "active" and address:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as error:
                raise CloudError(f"Lambda returned an invalid instance IP: {address!r}") from error
            if parsed_address.version != 4 or not parsed_address.is_global:
                raise CloudError(f"Lambda returned a non-public instance IP: {address!r}")
            return instance
        if status_value in {"unhealthy", "terminated", "preempted"}:
            raise CloudError(f"Instance entered terminal status: {status_value}")
        time.sleep(POLL_SECONDS)
    raise CloudError(f"Timed out waiting {timeout}s for the instance to become active.")


def wait_for_ssh(
    address: str, private_key: Path, known_hosts: Path, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    command = [
        "ssh",
        *ssh_options(private_key, known_hosts),
        f"ubuntu@{address}",
        "true",
    ]
    while time.monotonic() < deadline:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode == 0:
            print("SSH is ready.")
            return
        last_error = result.stderr.strip()
        time.sleep(POLL_SECONDS)
    raise CloudError(f"Timed out waiting for SSH. Last error: {last_error}")


def run_ssh(
    address: str,
    private_key: Path,
    known_hosts: Path,
    remote_command: str,
    *,
    log_path: Path | None = None,
) -> int:
    command = [
        "ssh",
        *ssh_options(private_key, known_hosts),
        f"ubuntu@{address}",
        remote_command,
    ]
    if log_path is None:
        return subprocess.run(command, check=False).returncode
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def retrieve(
    address: str,
    private_key: Path,
    known_hosts: Path,
    remote_dir: str,
    local_dir: Path,
) -> tuple[Path, Path]:
    archive_name = "raymma-cloud-results.tar.gz"
    digest_name = f"{archive_name}.sha256"
    remote_build = f"/home/ubuntu/{remote_dir}/build"
    command = [
        "scp",
        *ssh_options(private_key, known_hosts),
        f"ubuntu@{address}:{remote_build}/{archive_name}",
        f"ubuntu@{address}:{remote_build}/{digest_name}",
        str(local_dir),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise CloudError(f"Result download failed: {result.stderr.strip()}")
    return local_dir / archive_name, local_dir / digest_name


def verify_archive(archive: Path, digest_file: Path) -> str:
    fields = digest_file.read_text().strip().split()
    if len(fields) < 2 or fields[1].lstrip("*") != archive.name:
        raise CloudError("Downloaded SHA-256 sidecar has an unexpected format.")
    expected = fields[0].lower()
    actual_hash = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            actual_hash.update(chunk)
    actual = actual_hash.hexdigest()
    if actual != expected:
        raise CloudError(f"Archive checksum mismatch: expected {expected}, got {actual}")
    return actual


def terminate_instance(api: LambdaAPI, instance_id: str, timeout: int = 180) -> None:
    instance_id = validate_instance_id(instance_id)
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            api.request(
                "POST", "instance-operations/terminate", {"instance_ids": [instance_id]}
            )
            print(f"Termination requested through the Lambda API: {instance_id}")
        except APIHTTPError as error:
            if error.status == 404:
                print(f"Termination confirmed (instance absent): {instance_id}")
                return
            last_error = error
        except CloudError as error:
            last_error = error

        poll_deadline = min(deadline, time.monotonic() + 60)
        while time.monotonic() < poll_deadline:
            try:
                instance = api.request("GET", f"instances/{instance_id}")
            except APIHTTPError as error:
                if error.status == 404:
                    print(f"Termination confirmed: {instance_id}")
                    return
                last_error = error
                break
            except CloudError as error:
                last_error = error
                break
            status_value = str(instance.get("status", "unknown"))
            if status_value in {"terminated", "preempted"}:
                print(f"Termination confirmed ({status_value}): {instance_id}")
                return
            time.sleep(POLL_SECONDS)
        if time.monotonic() >= deadline:
            break
        if attempt < 2:
            print(f"Termination not yet confirmed; retrying request ({attempt + 2}/3).")
    raise CloudError(
        f"Could not confirm termination for {instance_id}: "
        f"{last_error or 'instance remained present'}"
    )


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if "token" in key.lower() or key.lower() == "jupyter_url"
            else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def inventory_command(args: argparse.Namespace) -> int:
    api = api_from_args(args)
    payload = {
        "gpu_capacity": available_cloud_gpus(api.request("GET", "instance-types")),
        "instances": api.request("GET", "instances"),
        "images": api.request("GET", "images"),
        "ssh_keys": api.request("GET", "ssh-keys"),
        "firewall_rulesets": api.request("GET", "firewall-rulesets"),
        "global_firewall": api.request("GET", "firewall-rulesets/global"),
    }
    if args.json:
        print(json.dumps(redact_sensitive(payload), indent=2, sort_keys=True))
        return 0
    print("Available supported single-GPU capacity:")
    for item in payload["gpu_capacity"]:
        price = item["price_cents_per_hour"] / 100
        print(f"  {item['name']}: ${price:.2f}/h; {', '.join(item['regions'])}")
    if not payload["gpu_capacity"]:
        print("  none")
    print("Running instances:")
    for instance in payload["instances"]:
        print(
            f"  {instance.get('id')}  {instance.get('status')}  "
            f"{instance.get('name', '')}  {instance.get('ip', '')}"
        )
    if not payload["instances"]:
        print("  none")
    print("SSH keys:", ", ".join(key.get("name", "") for key in payload["ssh_keys"]) or "none")
    print(
        "Firewall rulesets:",
        ", ".join(item.get("name", "") for item in payload["firewall_rulesets"])
        or "none",
    )
    return 0


def terminate_command(args: argparse.Namespace) -> int:
    api = api_from_args(args)
    terminate_instance(api, args.instance_id)
    return 0


def validate_local_commands() -> None:
    for command in ("ssh", "scp", "ssh-keygen"):
        if shutil.which(command) is None:
            raise CloudError(f"Required local command is unavailable: {command}")


def parse_ssh_cidr(value: str) -> str:
    try:
        network = ipaddress.ip_network(value, strict=False)
    except ValueError as error:
        raise CloudError(f"Invalid --ssh-cidr: {error}") from error
    special = (
        network.is_multicast
        or network.is_reserved
        or network.is_unspecified
        or network.is_loopback
        or network.is_link_local
        or network.is_private
    )
    if network.version != 4 or not network.is_global or special or network.prefixlen == 0:
        raise CloudError(
            "--ssh-cidr must be a public-unicast IPv4 address or bounded CIDR; "
            "0.0.0.0/0 is refused."
        )
    return str(network)


def run_command(args: argparse.Namespace) -> int:
    validate_local_commands()
    public_key, _, private_key = read_ssh_keys(args.ssh_public_key, args.ssh_private_key)
    cidr = parse_ssh_cidr(args.ssh_cidr)
    api = api_from_args(args)

    instance_types = api.request("GET", "instance-types")
    choice = choose_capacity(instance_types, args.instance_type, args.region)
    enforce_price_cap(choice, args.max_price_cents)
    images = api.request("GET", "images")
    validate_image(images, args.image_family, choice["region"])
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@+-]{0,199}", args.ref)
        or ".." in args.ref
        or "@{" in args.ref
    ):
        raise CloudError("--ref is not a safe Git branch, tag, or commit name.")
    if args.repository.startswith("-"):
        raise CloudError("--repository cannot begin with '-'.")

    cidr_hash = hashlib.sha256(cidr.encode()).hexdigest()[:10]
    firewall_name = args.firewall_name or f"raymma-{choice['region']}-{cidr_hash}"
    for label, value in (
        ("--ssh-key-name", args.ssh_key_name),
        ("--firewall-name", firewall_name),
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
            raise CloudError(
                f"{label} must be 1-64 characters using letters, digits, dot, "
                "dash, or underscore."
            )
    check_global_firewall(api, cidr, args.allow_broad_global_ssh)

    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = uuid.uuid4().hex
    instance_name = args.name or f"raymma-gpu-{run_stamp.lower()}-{run_id[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", instance_name):
        raise CloudError(
            "--name must be 1-64 characters using letters, digits, dot, dash, "
            "or underscore, and must start with a letter or digit."
        )
    recovery_plan = {
        "run_id": run_id,
        "name": instance_name,
        "instance_type": choice["name"],
        "region": choice["region"],
    }
    existing_instances = api.request("GET", "instances")
    if any(instance.get("name") == instance_name for instance in existing_instances):
        raise CloudError(f"An instance is already named {instance_name!r}.")
    if any(tag_value(instance, "run-id") == run_id for instance in existing_instances):
        raise CloudError("The random recovery tag already exists; rerun the command.")

    output_root = args.output_dir.expanduser().resolve()
    local_dir = output_root / instance_name
    if not args.dry_run and local_dir.exists():
        raise CloudError(f"Output directory already exists: {local_dir}")
    print(
        f"Plan: launch {choice['name']} in {choice['region']} at "
        f"${choice['price_cents_per_hour'] / 100:.2f}/h; image "
        f"{args.image_family}; SSH source {cidr}; no filesystem."
    )
    if not args.dry_run and not args.yes:
        try:
            confirmation = input("Type 'launch' to start billing: ").strip()
        except EOFError as error:
            raise CloudError("Non-interactive use requires --yes.") from error
        if confirmation != "launch":
            raise CloudError("Launch cancelled.")

    if not args.dry_run:
        local_dir.mkdir(parents=True, exist_ok=False)
        (local_dir / "recovery-plan.json").write_text(
            json.dumps(recovery_plan, indent=2, sort_keys=True) + "\n"
        )

    ssh_key_id = ensure_ssh_key(api, args.ssh_key_name, public_key, args.dry_run)
    firewall_id = ensure_firewall_ruleset(
        api, firewall_name, choice["region"], cidr, args.dry_run
    )
    launch_body = {
        "region_name": choice["region"],
        "instance_type_name": choice["name"],
        "ssh_key_names": [args.ssh_key_name],
        "file_system_names": [],
        "name": instance_name,
        "image": {"family": args.image_family},
        "tags": [
            {"key": "project", "value": "raymma"},
            {"key": "run-id", "value": run_id},
        ],
        "firewall_rulesets": [{"id": firewall_id}],
    }
    if args.dry_run:
        print("DRY RUN: no key, firewall, instance, SSH, or local output was mutated.")
        print(json.dumps(launch_body, indent=2, sort_keys=True))
        return 0

    known_hosts = local_dir / "known_hosts"
    (local_dir / "launch-plan.json").write_text(
        json.dumps(
            {
                "instance": launch_body,
                "price_cents_per_hour": choice["price_cents_per_hour"],
                "ssh_key_id": ssh_key_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    instance_id: str | None = None
    known_instance_ids: set[str] = set()
    address = ""
    remote_dir: str | None = None
    archive_verified = False
    retrieval_error: BaseException | None = None
    launch_recovery_needed = False
    termination_confirmed = False
    start = time.monotonic()
    job_error: BaseException | None = None
    try:
        launch_recovery_needed = True
        try:
            instance_id = launch_instance(api, launch_body, recovery_plan)
        except DefinitiveLaunchError:
            launch_recovery_needed = False
            raise
        except MultipleLaunchError as error:
            known_instance_ids.update(error.instance_ids)
            launch_recovery_needed = False
            (local_dir / "unexpected-instance-ids.json").write_text(
                json.dumps(error.instance_ids, indent=2) + "\n"
            )
            raise
        launch_recovery_needed = False
        known_instance_ids.add(instance_id)
        (local_dir / "instance-id.txt").write_text(instance_id + "\n")
        print(f"Launched instance: {instance_id}")
        remote_dir = f"RayMMA-{instance_id[:12]}"
        instance = wait_for_instance(api, instance_id, args.launch_timeout)
        address = str(instance["ip"])
        wait_for_ssh(address, private_key, known_hosts, args.ssh_timeout)

        if not args.skip_bootstrap:
            bootstrap = (
                "sudo apt-get update && "
                "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "build-essential cmake git python3 ca-certificates"
            )
            if run_ssh(
                address,
                private_key,
                known_hosts,
                bootstrap,
                log_path=local_dir / "bootstrap.log",
            ):
                raise CloudError("Remote package bootstrap failed.")

        clone = " && ".join(
            [
                f"git init {shlex.quote(remote_dir)}",
                f"cd {shlex.quote(remote_dir)}",
                f"git remote add origin {shlex.quote(args.repository)}",
                f"git fetch --depth 1 -- origin {shlex.quote(args.ref)}",
                "git checkout --detach FETCH_HEAD",
            ]
        )
        if run_ssh(
            address,
            private_key,
            known_hosts,
            clone,
            log_path=local_dir / "checkout.log",
        ):
            raise CloudError("Remote Git checkout failed.")

        benchmark = (
            f"cd {shlex.quote(remote_dir)} && "
            f"./tools/run_cloud_gpu.sh --profile {shlex.quote(args.profile)}"
        )
        benchmark_status = run_ssh(
            address,
            private_key,
            known_hosts,
            benchmark,
            log_path=local_dir / "remote-run.log",
        )
        try:
            archive, digest = retrieve(
                address, private_key, known_hosts, remote_dir, local_dir
            )
            verified = verify_archive(archive, digest)
            archive_verified = True
            print(f"Downloaded and verified: {archive} ({verified})")
        except CloudError as error:
            retrieval_error = error
        if benchmark_status:
            raise CloudError(
                f"Remote runner exited {benchmark_status}; "
                + ("the evidence archive was verified." if archive_verified else "retrieval will be retried before termination.")
            )
        if retrieval_error:
            raise retrieval_error
    except BaseException as error:  # termination must also run for Ctrl-C
        job_error = error
    finally:
        if launch_recovery_needed and not known_instance_ids:
            try:
                recovered = recover_launch(api, recovery_plan)
                if recovered:
                    instance_id = validate_instance_id(str(recovered.get("id", "")))
                    known_instance_ids.add(instance_id)
                    (local_dir / "instance-id.txt").write_text(instance_id + "\n")
                    print(f"Recovered interrupted launch by run tag: {instance_id}")
            except BaseException as recovery_error:
                print(
                    f"URGENT: launch recovery failed: {recovery_error}",
                    file=sys.stderr,
                )
            if not known_instance_ids:
                warning = (
                    "URGENT: launch outcome remains unknown. Search inventory for "
                    f"run-id {run_id} or name {instance_name}; terminate any match."
                )
                print(warning, file=sys.stderr)
                (local_dir / "UNKNOWN-LAUNCH.txt").write_text(warning + "\n")

        if instance_id and address and remote_dir and not archive_verified:
            for attempt, delay in enumerate((2, 5, 10), start=1):
                time.sleep(delay)
                try:
                    archive, digest = retrieve(
                        address, private_key, known_hosts, remote_dir, local_dir
                    )
                    verified = verify_archive(archive, digest)
                    archive_verified = True
                    print(
                        f"Best-effort retrieval succeeded: {archive} ({verified})"
                    )
                    if job_error is retrieval_error:
                        job_error = None
                    break
                except BaseException as error:
                    print(
                        f"Result retrieval attempt {attempt}/3 failed: {error}",
                        file=sys.stderr,
                    )

        force_terminate = len(known_instance_ids) > 1
        if known_instance_ids and (not args.keep_instance or force_terminate):
            termination_confirmed = True
            for known_id in sorted(known_instance_ids):
                try:
                    terminate_instance(api, known_id)
                except BaseException as terminate_error:
                    termination_confirmed = False
                    print(
                        f"URGENT: automatic termination failed: {terminate_error}\n"
                        f"Run: {shlex.quote(sys.executable)} tools/lambda_cloud.py "
                        f"terminate --instance-id {shlex.quote(known_id)}",
                        file=sys.stderr,
                    )
                    if job_error is None:
                        job_error = terminate_error
        elif known_instance_ids:
            print(
                f"Instance kept by request and is still billing: {instance_id}\n"
                f"Terminate with: {sys.executable} tools/lambda_cloud.py "
                f"terminate --instance-id {instance_id}"
            )

    elapsed = time.monotonic() - start
    if known_instance_ids:
        billed_minutes = max(1, math.ceil(elapsed / 60))
        estimated = (
            choice["price_cents_per_hour"]
            / 100
            * billed_minutes
            / 60
            * len(known_instance_ids)
        )
        qualifier = (
            "through confirmed termination"
            if termination_confirmed
            else "incurred so far; billing may continue"
        )
        print(
            f"Elapsed {elapsed / 60:.1f} min; estimated compute ${estimated:.2f} "
            f"{qualifier}, before tax."
        )
    elif launch_recovery_needed:
        print("Launch outcome and cost remain unknown; inspect inventory immediately.")
    if job_error is not None:
        raise job_error
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate a RayMMA run through the official Lambda Cloud REST API."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="read-only capacity and resource list")
    inventory.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    inventory.set_defaults(func=inventory_command)

    terminate = subparsers.add_parser("terminate", help="rescue termination by instance ID")
    terminate.add_argument("--instance-id", required=True)
    terminate.set_defaults(func=terminate_command)

    run = subparsers.add_parser("run", help="launch, benchmark, download, and terminate")
    run.add_argument("--ssh-public-key", type=Path, required=True)
    run.add_argument("--ssh-private-key", type=Path, required=True)
    run.add_argument("--ssh-cidr", required=True, help="public IPv4 address/CIDR, ideally /32")
    run.add_argument("--ssh-key-name", default="raymma")
    run.add_argument("--firewall-name")
    run.add_argument(
        "--allow-broad-global-ssh",
        action="store_true",
        help="accept workspace-global SSH sources outside --ssh-cidr",
    )
    run.add_argument(
        "--instance-type",
        help="exact live supported single-GPU type; auto-select B200 only",
    )
    run.add_argument(
        "--max-price-cents",
        type=int,
        help="refuse launch when the live hourly price exceeds this many cents",
    )
    run.add_argument("--region", help="exact region; auto-select from live capacity by default")
    run.add_argument("--image-family", default=DEFAULT_IMAGE_FAMILY)
    run.add_argument("--repository", default=DEFAULT_REPOSITORY)
    run.add_argument("--ref", default="main", help="public branch, tag, or commit to fetch")
    run.add_argument("--profile", choices=("quick", "archive"), default="quick")
    run.add_argument("--output-dir", type=Path, default=Path("lambda-results"))
    run.add_argument("--name", help="unique Lambda instance name")
    run.add_argument("--launch-timeout", type=int, default=900)
    run.add_argument("--ssh-timeout", type=int, default=600)
    run.add_argument("--skip-bootstrap", action="store_true")
    run.add_argument("--keep-instance", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="perform GETs and show plan only")
    run.add_argument("--yes", action="store_true", help="confirm billable launch non-interactively")
    run.set_defaults(func=run_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except CloudError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

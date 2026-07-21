#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline safety tests for tools/lambda_cloud.py."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lambda_cloud as cloud


def instance_types(regions: tuple[str, ...] = ("us-west-1",)) -> dict:
    return {
        "gpu_1x_a100": {
            "instance_type": {
                "name": "gpu_1x_a100",
                "description": "1x NVIDIA A100",
                "gpu_description": "NVIDIA A100 40 GB",
                "price_cents_per_hour": 199,
                "specs": {"gpus": 1},
            },
            "regions_with_capacity_available": [
                {"name": region} for region in regions
            ],
        },
        "gpu_8x_a100": {
            "instance_type": {
                "name": "gpu_8x_a100",
                "description": "8x NVIDIA A100",
                "price_cents_per_hour": 1592,
                "specs": {"gpus": 8},
            },
            "regions_with_capacity_available": [{"name": "us-west-1"}],
        },
    }


def planned_instance(plan: dict[str, str], instance_id: str = "0123456789abcdef") -> dict:
    return {
        "id": instance_id,
        "name": plan["name"],
        "status": "active",
        "ip": "8.8.8.8",
        "instance_type": {"name": plan["instance_type"]},
        "region": {"name": plan["region"]},
        "tags": [
            {"key": "project", "value": "raymma"},
            {"key": "run-id", "value": plan["run_id"]},
        ],
    }


class FakeAPI:
    def __init__(self, responses: dict[tuple[str, str], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, object]] = []

    def request(self, method: str, path: str, body=None):
        self.calls.append((method, path, body))
        value = self.responses[(method, path)]
        if isinstance(value, ResponseQueue):
            value = value.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class ResponseQueue:
    def __init__(self, *values: object) -> None:
        self.values = list(values)


class LambdaCloudTests(unittest.TestCase):
    def test_supported_gpu_priority_and_exact_model_tokens(self) -> None:
        types = instance_types()
        types.update(
            {
                "gpu_1x_a10": {
                    "instance_type": {
                        "name": "gpu_1x_a10",
                        "description": "1x NVIDIA A10",
                        "price_cents_per_hour": 129,
                        "specs": {"gpus": 1},
                    },
                    "regions_with_capacity_available": [{"name": "us-west-1"}],
                },
                "gpu_1x_h100": {
                    "instance_type": {
                        "name": "gpu_1x_h100",
                        "description": "1x NVIDIA H100",
                        "price_cents_per_hour": 299,
                        "specs": {"gpus": 1},
                    },
                    "regions_with_capacity_available": [{"name": "us-west-1"}],
                },
            }
        )
        choices = cloud.available_cloud_gpus(types)
        self.assertEqual(
            [item["model"] for item in choices], ["a100"]
        )
        self.assertNotIn("h100", [item["model"] for item in choices])
        self.assertNotIn("a10", [item["model"] for item in choices])
        self.assertEqual(cloud.choose_capacity(types, None, None)["model"], "a100")

    def test_selects_only_one_a100_and_excludes_unprotected_region(self) -> None:
        choices = cloud.available_a100s(instance_types(("us-south-1", "us-west-1")))
        self.assertEqual([item["name"] for item in choices], ["gpu_1x_a100"])
        self.assertEqual(choices[0]["regions"], ["us-west-1"])
        with self.assertRaises(cloud.CloudError):
            cloud.choose_capacity(instance_types(("us-south-1",)), None, None)

    def test_cidr_requires_public_unicast_and_bounded_network(self) -> None:
        self.assertEqual(cloud.parse_ssh_cidr("8.8.8.8/32"), "8.8.8.8/32")
        for value in (
            "not-an-ip",
            "0.0.0.0/0",
            "10.0.0.1/32",
            "127.0.0.1/32",
            "224.0.0.1/32",
            "2001:4860:4860::8888/128",
        ):
            with self.subTest(value=value), self.assertRaises(cloud.CloudError):
                cloud.parse_ssh_cidr(value)

    def test_global_firewall_fails_closed(self) -> None:
        broad = FakeAPI(
            {
                ("GET", "firewall-rulesets/global"): {
                    "rules": [
                        {
                            "protocol": "tcp",
                            "port_range": [22, 22],
                            "source_network": "0.0.0.0/0",
                        }
                    ]
                }
            }
        )
        with self.assertRaises(cloud.CloudError):
            cloud.check_global_firewall(broad, "8.8.8.8/32", False)
        with patch("sys.stderr", new_callable=io.StringIO):
            cloud.check_global_firewall(broad, "8.8.8.8/32", True)

    def test_existing_ssh_key_mismatch_is_rejected(self) -> None:
        api = FakeAPI(
            {
                ("GET", "ssh-keys"): [
                    {
                        "id": "key-id",
                        "name": "raymma",
                        "public_key": "ssh-ed25519 AAAAremote",
                    }
                ]
            }
        )
        with self.assertRaises(cloud.CloudError):
            cloud.ensure_ssh_key(api, "raymma", "ssh-ed25519 AAAAlocal", False)

    def test_dry_run_has_no_post_and_creates_no_output(self) -> None:
        api = FakeAPI(
            {
                ("GET", "instance-types"): instance_types(),
                ("GET", "images"): [
                    {
                        "family": cloud.DEFAULT_IMAGE_FAMILY,
                        "architecture": "x86_64",
                        "region": {"name": "us-west-1"},
                    }
                ],
                ("GET", "firewall-rulesets/global"): {"rules": []},
                ("GET", "instances"): [],
                ("GET", "ssh-keys"): [],
                ("GET", "firewall-rulesets"): [],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            args = cloud.build_parser().parse_args(
                [
                    "run",
                    "--ssh-public-key",
                    "public",
                    "--ssh-private-key",
                    "private",
                    "--ssh-cidr",
                    "8.8.8.8/32",
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ]
            )
            with (
                patch.object(cloud, "api_from_args", return_value=api),
                patch.object(cloud, "validate_local_commands"),
                patch.object(
                    cloud,
                    "read_ssh_keys",
                    return_value=("ssh-ed25519 AAAAtest", Path("public"), Path("private")),
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(cloud.run_command(args), 0)
            self.assertFalse(output.exists())
            self.assertFalse(any(method == "POST" for method, _, _ in api.calls))

    def test_launch_recovers_only_by_full_random_tag_plan(self) -> None:
        plan = {
            "run_id": "a" * 32,
            "name": "raymma-a100-test",
            "instance_type": "gpu_1x_a100",
            "region": "us-west-1",
        }
        api = FakeAPI(
            {
                ("POST", "instance-operations/launch"): cloud.APITransportError("lost"),
                ("GET", "instances"): [planned_instance(plan)],
            }
        )
        with (
            patch.object(cloud.time, "sleep"),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            result = cloud.launch_instance(api, {"name": plan["name"]}, plan)
        self.assertEqual(result, "0123456789abcdef")

        rejected = FakeAPI(
            {
                ("POST", "instance-operations/launch"): cloud.APIHTTPError(
                    400, "invalid", "bad request"
                )
            }
        )
        with self.assertRaises(cloud.DefinitiveLaunchError):
            cloud.launch_instance(rejected, {"name": plan["name"]}, plan)
        self.assertEqual(len(rejected.calls), 1)

        unrelated = planned_instance({**plan, "run_id": "b" * 32})
        same_name = FakeAPI(
            {
                ("POST", "instance-operations/launch"): cloud.APITransportError("lost"),
                ("GET", "instances"): [unrelated],
            }
        )
        with (
            patch.object(cloud.time, "sleep"),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            with self.assertRaises(cloud.AmbiguousLaunchError):
                cloud.launch_instance(same_name, {"name": plan["name"]}, plan)

    def test_archive_digest_and_inventory_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "raymma-a100-results.tar.gz"
            archive.write_bytes(b"evidence")
            expected = hashlib.sha256(b"evidence").hexdigest()
            digest = archive.with_suffix(archive.suffix + ".sha256")
            digest.write_text(f"{expected}  {archive.name}\n")
            self.assertEqual(cloud.verify_archive(archive, digest), expected)
        redacted = cloud.redact_sensitive(
            {"jupyter_token": "secret", "jupyter_url": "https://x/?token=secret"}
        )
        self.assertEqual(redacted["jupyter_token"], "[redacted]")
        self.assertEqual(redacted["jupyter_url"], "[redacted]")

    def test_post_launch_interrupt_still_terminates(self) -> None:
        instance_id = "0123456789abcdef"
        api = FakeAPI(
            {
                ("GET", "instance-types"): instance_types(),
                ("GET", "images"): [
                    {
                        "family": cloud.DEFAULT_IMAGE_FAMILY,
                        "architecture": "x86_64",
                        "region": {"name": "us-west-1"},
                    }
                ],
                ("GET", "firewall-rulesets/global"): {"rules": []},
                ("GET", "instances"): [],
                ("GET", "ssh-keys"): [],
                ("POST", "ssh-keys"): {"id": "ssh-id"},
                ("GET", "firewall-rulesets"): [],
                ("POST", "firewall-rulesets"): {"id": "firewall-id"},
                ("POST", "instance-operations/launch"): {
                    "instance_ids": [instance_id]
                },
                ("GET", f"instances/{instance_id}"): ResponseQueue(
                    {"id": instance_id, "status": "active", "ip": "8.8.8.8"},
                    cloud.APIHTTPError(404, "not-found", "gone"),
                ),
                ("POST", "instance-operations/terminate"): {},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            args = cloud.build_parser().parse_args(
                [
                    "run",
                    "--ssh-public-key",
                    "public",
                    "--ssh-private-key",
                    "private",
                    "--ssh-cidr",
                    "8.8.8.8/32",
                    "--output-dir",
                    temporary,
                    "--yes",
                ]
            )
            with (
                patch.object(cloud, "api_from_args", return_value=api),
                patch.object(cloud, "validate_local_commands"),
                patch.object(
                    cloud,
                    "read_ssh_keys",
                    return_value=("ssh-ed25519 AAAAtest", Path("public"), Path("private")),
                ),
                patch.object(cloud, "wait_for_ssh"),
                patch.object(cloud, "run_ssh", side_effect=[0, 0, KeyboardInterrupt()]),
                patch.object(cloud, "retrieve", side_effect=cloud.CloudError("not ready")),
                patch.object(cloud.time, "sleep"),
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    cloud.run_command(args)
        self.assertTrue(
            any(
                method == "POST" and path == "instance-operations/terminate"
                for method, path, _ in api.calls
            )
        )


if __name__ == "__main__":
    unittest.main()

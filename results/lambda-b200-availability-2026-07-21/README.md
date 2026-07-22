# Lambda Cloud B200 overnight availability watch

From July 21, 2026 at 18:58:51 NDT (21:28:51 UTC) through July 22 at
06:58:51 NDT (09:28:51 UTC), RayMMA's unattended Lambda helper watched for
one eligible `gpu_1x_b200_sxm6` rental. It made 139 successful API inventory
checks during the 12-hour wall-clock window. No single-GPU B200 capacity
appeared, so the helper launched no instance, incurred no B200 compute charge,
and produced no B200 benchmark result.

This is an availability record, not performance evidence. The randomized
three-to-seven-minute polling schedule included one longer gap while the local
laptop was suspended; “12-hour watch” therefore describes the bounded
wall-clock window rather than an uninterrupted request stream.

The watcher was restricted to:

- one B200 only, with no A100, H100, or A10 fallback;
- a hard live-price ceiling of $7.00/hour;
- an archive run only after capacity appeared; and
- automatic result retrieval and termination after a launch.

At expiry the systemd service returned success, its final state was
`status=expired`, and a separate live inventory check again reported no
running instances.

See [`tools/watch_lambda_b200.sh`](../../tools/watch_lambda_b200.sh) for the
executed policy and [the Lambda workflow](../../docs/LAMBDA_CLOUD.md) for its
safety model.

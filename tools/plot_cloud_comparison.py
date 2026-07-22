#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Thomas Butyn
# SPDX-License-Identifier: MIT
"""Build the published A10/A100/H100 RayMMA comparison chart."""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
import shutil
import statistics
import subprocess


GPUS = (
    ("A10", "a10", "#2563eb"),
    ("A100", "a100", "#0f8f83"),
    ("H100", "h100", "#ea580c"),
)
LEAVES = (4, 8, 16, 32, 64, 128, 256)
SVG_WIDTH = 1600
SVG_HEIGHT = 900
COMPARABILITY_PATHS = (
    "CMakeLists.txt",
    "src/production_bvh.h",
    "src/research_benchmark.cu",
    "tools/run_cloud_gpu.sh",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--svg",
        type=Path,
        default=root / "docs/assets/cloud-gpu-crossover.svg",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results/cloud-gpu-comparison-2026-07-21.csv",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Also rasterize with ImageMagick when convert or magick is installed.",
    )
    return parser.parse_args()


def validate_comparability(root: Path) -> None:
    reference: dict[str, str] | None = None
    for label, slug, _color in GPUS:
        manifest = root / f"results/lambda-{slug}-2026-07-21/source-sha256.txt"
        hashes = {
            line.split(maxsplit=1)[1].removeprefix("./"): line.split(maxsplit=1)[0]
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if len(line.split(maxsplit=1)) == 2
        }
        selected = {name: hashes[name] for name in COMPARABILITY_PATHS}
        if reference is None:
            reference = selected
        elif selected != reference:
            raise ValueError(f"{label} benchmark source hashes do not match")


def load_series(root: Path) -> dict[str, list[dict[str, float | int]]]:
    series: dict[str, list[dict[str, float | int]]] = {}
    for label, slug, _color in GPUS:
        path = (
            root
            / f"results/lambda-{slug}-2026-07-21/grid-primary-uvt-depthsorted.csv"
        )
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))

        points: list[dict[str, float | int]] = []
        for leaf in LEAVES:
            samples: dict[str, list[float]] = {}
            for scope in ("integrated-cuda32", "integrated-tensor"):
                samples[scope] = [
                    float(row["milliseconds"])
                    for row in rows
                    if row["scene"] == "Grid"
                    and row["ray_kind"] == "primary"
                    and row["ray_order"] == "coherent"
                    and row["bvh_builder"] == "builtin-binned-SAH"
                    and row["tensor_variant"] == "uvt-depthsorted"
                    and row["width"] == "256"
                    and row["height"] == "144"
                    and int(row["max_leaf_triangles"]) == leaf
                    and row["timing_scope"] == scope
                ]
                if len(samples[scope]) != 9:
                    raise ValueError(
                        f"{path}: expected 9 {scope} samples at leaf {leaf}, "
                        f"found {len(samples[scope])}"
                    )

            cuda_ms = statistics.median(samples["integrated-cuda32"])
            tensor_ms = statistics.median(samples["integrated-tensor"])
            points.append(
                {
                    "leaf": leaf,
                    "cuda_ms": cuda_ms,
                    "tensor_ms": tensor_ms,
                    "speedup": cuda_ms / tensor_ms,
                }
            )
        series[label] = points
    return series


def write_csv(path: Path, series: dict[str, list[dict[str, float | int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "gpu",
                "scene",
                "ray_kind",
                "ray_order",
                "bvh_builder",
                "tensor_variant",
                "width",
                "height",
                "max_leaf_triangles",
                "cuda32_median_ms",
                "tensor_median_ms",
                "cuda32_over_tensor_speedup",
                "samples_per_scope",
            )
        )
        for label, _slug, _color in GPUS:
            for point in series[label]:
                writer.writerow(
                    (
                        label,
                        "Grid",
                        "primary",
                        "coherent",
                        "builtin-binned-SAH",
                        "uvt-depthsorted",
                        256,
                        144,
                        point["leaf"],
                        f'{point["cuda_ms"]:.6f}',
                        f'{point["tensor_ms"]:.6f}',
                        f'{point["speedup"]:.6f}',
                        9,
                    )
                )


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    fill: str = "#172033",
    weight: int = 400,
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" {extra}>'
        f"{escape(value)}</text>"
    )


def write_svg(path: Path, series: dict[str, list[dict[str, float | int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" '
        f'height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        "<title>RayMMA cloud GPU crossover comparison</title>",
        "<desc>Line chart of approximate Tensor speedup over same-tree CUDA32 "
        "by BVH leaf size on A10, A100, and H100, plus absolute latency at leaf 128.</desc>",
        '<rect width="1600" height="900" fill="#f8fafc"/>',
        '<style>text { font-family: "DejaVu Sans", Arial, sans-serif; }</style>',
        text(70, 66, "When dense BVH leaves favor Tensor Cores", size=42, weight=700),
        text(
            70,
            108,
            "Same 32,768-triangle Grid workload on Lambda Cloud · coherent primary rays · 9-sample medians",
            size=21,
            fill="#526079",
        ),
    ]

    # Left panel: within-GPU crossover.
    px, py, pw, ph = 70, 150, 930, 610
    cx0, cy0, cw, ch = 142, 235, 816, 438
    parts.extend(
        (
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="18" fill="#ffffff" stroke="#dce3ec"/>',
            text(px + 30, py + 47, "Tensor speedup over same-tree CUDA32", size=25, weight=700),
            text(px + 30, py + 77, "CUDA32 median ÷ Tensor median · above 1.0× means Tensor is faster", size=17, fill="#657087"),
        )
    )

    y_min, y_max = 0.35, 1.80

    def x_for(index: int) -> float:
        return cx0 + index * cw / (len(LEAVES) - 1)

    def y_for(value: float) -> float:
        return cy0 + (y_max - value) * ch / (y_max - y_min)

    crossover_x = (x_for(3) + x_for(4)) / 2
    parts.append(
        f'<rect x="{crossover_x:.1f}" y="{cy0}" width="{cx0 + cw - crossover_x:.1f}" '
        f'height="{ch}" fill="#ecfdf5" stroke="none"/>'
    )
    for tick in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        y = y_for(tick)
        width = 2.5 if tick == 1.0 else 1
        color = "#475569" if tick == 1.0 else "#dfe5ec"
        parts.append(
            f'<line x1="{cx0}" y1="{y:.1f}" x2="{cx0 + cw}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )
        parts.append(text(cx0 - 15, y + 7, f"{tick:.2g}×", size=16, fill="#667085", anchor="end"))
    parts.append(text(cx0 + cw - 10, y_for(1.0) - 13, "Tensor faster", size=16, fill="#13795b", weight=700, anchor="end"))
    parts.append(text(cx0 + 10, y_for(1.0) + 25, "CUDA32 faster", size=16, fill="#596579", weight=700))

    for index, leaf in enumerate(LEAVES):
        x = x_for(index)
        parts.append(f'<line x1="{x:.1f}" y1="{cy0}" x2="{x:.1f}" y2="{cy0 + ch}" stroke="#eef1f5"/>')
        parts.append(text(x, cy0 + ch + 31, str(leaf), size=17, fill="#526079", anchor="middle"))
    parts.append(text(cx0 + cw / 2, cy0 + ch + 64, "Maximum triangles per BVH leaf", size=18, fill="#526079", anchor="middle"))

    for label, _slug, color in GPUS:
        points = series[label]
        coords = " ".join(
            f'{x_for(index):.1f},{y_for(float(point["speedup"])):.1f}'
            for index, point in enumerate(points)
        )
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>')
        for index, point in enumerate(points):
            parts.append(
                f'<circle cx="{x_for(index):.1f}" cy="{y_for(float(point["speedup"])):.1f}" '
                f'r="6" fill="#ffffff" stroke="{color}" stroke-width="4"/>'
            )
        peak = points[5]
        parts.append(
            text(
                x_for(5),
                y_for(float(peak["speedup"])) - 14,
                f'{float(peak["speedup"]):.2f}×',
                size=16,
                fill=color,
                weight=700,
                anchor="middle",
            )
        )

    legend_y = cy0 + 30
    for index, (label, _slug, color) in enumerate(GPUS):
        lx = cx0 + 18 + index * 145
        parts.append(f'<line x1="{lx}" y1="{legend_y - 6}" x2="{lx + 35}" y2="{legend_y - 6}" stroke="{color}" stroke-width="5"/>')
        parts.append(text(lx + 48, legend_y, label, size=18, weight=700))

    # Right panel: absolute latency at the common peak leaf size.
    rx, ry, rw, rh = 1030, 150, 500, 610
    parts.extend(
        (
            f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="18" fill="#ffffff" stroke="#dce3ec"/>',
            text(rx + 30, ry + 47, "Absolute latency at leaf 128", size=25, weight=700),
            text(rx + 30, ry + 77, "this workload · milliseconds · lower is better", size=16, fill="#657087"),
        )
    )
    bx0, bw = rx + 112, 325
    max_ms = 0.95
    for tick in (0.0, 0.25, 0.5, 0.75):
        x = bx0 + tick / max_ms * bw
        parts.append(f'<line x1="{x:.1f}" y1="{ry + 115}" x2="{x:.1f}" y2="{ry + 447}" stroke="#e8ecf1"/>')
        parts.append(text(x, ry + 474, f"{tick:.2f}", size=15, fill="#687386", anchor="middle"))

    for index, (label, _slug, color) in enumerate(GPUS):
        point = series[label][5]
        group_y = ry + 137 + index * 103
        parts.append(text(rx + 30, group_y + 31, label, size=19, weight=700))
        for offset, key, fill, stroke in (
            (0, "cuda_ms", "#e4e9f0", "#aab4c3"),
            (39, "tensor_ms", color, color),
        ):
            value = float(point[key])
            width = value / max_ms * bw
            parts.append(
                f'<rect x="{bx0}" y="{group_y + offset}" width="{width:.1f}" height="27" '
                f'rx="5" fill="{fill}" stroke="{stroke}"/>'
            )
            parts.append(text(bx0 + width + 9, group_y + offset + 20, f"{value:.3f}", size=15, fill="#354052", weight=700))

    legend_y = ry + 511
    parts.append(f'<rect x="{rx + 30}" y="{legend_y}" width="25" height="18" rx="3" fill="#e4e9f0" stroke="#aab4c3"/>')
    parts.append(text(rx + 65, legend_y + 16, "CUDA32", size=16, fill="#4a5568"))
    for offset, color in enumerate(("#2563eb", "#0f8f83", "#ea580c")):
        parts.append(
            f'<rect x="{rx + 190 + offset * 8}" y="{legend_y}" width="9" '
            f'height="18" fill="{color}"/>'
        )
    parts.append(text(rx + 225, legend_y + 16, "Tensor-owned", size=16, fill="#4a5568"))
    parts.append(
        f'<rect x="{rx + 30}" y="{ry + 542}" width="{rw - 60}" height="50" rx="8" fill="#fff7ed" stroke="#fed7aa"/>'
    )
    parts.append(text(rx + rw / 2, ry + 562, "Exact hybrid @ leaf 128: A10 1.07× · A100 1.02× · H100 0.97×", size=13, fill="#7c2d12", weight=700, anchor="middle"))
    parts.append(text(rx + rw / 2, ry + 583, "Lambda bill: $0.32 total · A10 8¢ · A100 5¢ · H100 19¢", size=14, fill="#9a3412", weight=700, anchor="middle"))

    parts.extend(
        (
            text(70, 802, "Approximate path shown: FP16-input uvt-depthsorted; Tensor output owns hit and depth; no Möller validation.", size=17, fill="#3e4a5e", weight=700),
            text(70, 829, "Plotted-primary maxima: 0.245% missed hits · 0.276% wrong closest triangles · 7.73% relative depth error.", size=16, fill="#596579"),
            text(70, 856, "At selective leaves, CUDA wins. Dense-leaf speedups are same-tree comparisons—not wins over the best BVH or RT Cores.", size=16, fill="#596579"),
            text(70, 884, "Source, raw samples, correctness counters, and plotting script: github.com/tabutyn/RayMMA", size=15, fill="#2563eb"),
            text(1530, 884, "B200: 139 checks / 12 h / no capacity", size=15, fill="#596579", weight=700, anchor="end"),
            "</svg>",
        )
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def rasterize(svg: Path, png: Path) -> None:
    executable = shutil.which("magick") or shutil.which("convert")
    if executable is None:
        raise RuntimeError("--png requires ImageMagick's magick or convert command")
    png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            executable,
            "-background",
            "white",
            "-density",
            "144",
            str(svg),
            "-resize",
            f"{SVG_WIDTH}x{SVG_HEIGHT}!",
            "-depth",
            "8",
            str(png),
        ),
        check=True,
    )


def main() -> int:
    args = parse_args()
    validate_comparability(args.root)
    series = load_series(args.root)
    write_csv(args.csv, series)
    write_svg(args.svg, series)
    if args.png is not None:
        rasterize(args.svg, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

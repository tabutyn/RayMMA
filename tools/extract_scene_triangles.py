# SPDX-FileCopyrightText: 2026 Thomas Butyn
# SPDX-License-Identifier: MIT
"""Blender-side glTF to minimal ray-tracer triangle cache converter."""
import argparse
import struct
import sys

import bpy
from mathutils import Vector


def parse_args():
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture-output", required=True)
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=0,
        help="Approximate triangle target; zero preserves the source mesh",
    )
    return parser.parse_args(args)


def triangle_count(objects):
    count = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        count += len(mesh.loop_triangles)
        evaluated.to_mesh_clear()
    return count


def decimate(objects, target_triangles):
    source_triangles = triangle_count(objects)
    if target_triangles <= 0 or target_triangles >= source_triangles:
        return source_triangles
    ratio = target_triangles / source_triangles
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new(name="RayMMA benchmark LOD", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.select_set(False)
    return source_triangles


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name,
    )
    if not objects:
        raise RuntimeError("The input scene contains no mesh objects")
    source_triangles = decimate(objects, args.target_triangles)

    records = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        uv_data = mesh.uv_layers.active.data if mesh.uv_layers.active else None
        for triangle in mesh.loop_triangles:
            points = [obj.matrix_world @ mesh.vertices[i].co for i in triangle.vertices]
            uvs = (
                [tuple(uv_data[i].uv) for i in triangle.loops]
                if uv_data
                else [(0.0, 0.0)] * 3
            )
            color = (196, 185, 160, 255)
            packed_color = (
                color[0] | color[1] << 8 | color[2] << 16 | color[3] << 24
            )
            records.append((points, uvs, packed_color))
        evaluated.to_mesh_clear()

    points = [point for triangle, _, _ in records for point in triangle]
    lo = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    hi = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    center = (lo + hi) * 0.5
    scale = 2.4 / max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)

    with open(args.output, "wb") as output:
        output.write(struct.pack("<8sI", b"BRTRI003", len(records)))
        for triangle, uvs, color in records:
            vertices = []
            for point in triangle:
                transformed = (point - center) * scale
                vertices.append(
                    Vector((transformed.x, transformed.z, -transformed.y))
                )
            converted = [
                *vertices[0],
                *(vertices[1] - vertices[0]),
                *(vertices[2] - vertices[0]),
            ]
            flattened_uvs = [value for uv in uvs for value in uv]
            output.write(struct.pack("<15fI", *converted, *flattened_uvs, color))

    base_color = None
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is None:
            continue
        socket = principled.inputs.get("Base Color")
        if socket and socket.is_linked:
            source = socket.links[0].from_node
            if source.type == "TEX_IMAGE":
                base_color = source.image
                break
    if base_color is None:
        raise RuntimeError("No base-color image found in the glTF asset")
    base_color.filepath_raw = args.texture_output
    base_color.file_format = "PNG"
    base_color.save()
    print(
        f"{args.input}: {source_triangles} source -> {len(records)} triangles "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()

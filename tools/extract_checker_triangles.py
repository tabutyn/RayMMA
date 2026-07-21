# SPDX-License-Identifier: MIT
"""Blender-side GLB to minimal ray-tracer triangle cache converter."""
import argparse
import struct
import bpy
from mathutils import Vector


def parse_args():
    args = []
    if "--" in __import__("sys").argv:
        args = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture-output", required=True)
    return parser.parse_args(args)


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)
    records = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted((o for o in bpy.context.scene.objects if o.type == "MESH"),
                      key=lambda o: o.name):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        color = (54, 38, 30, 255) if "black" in obj.name.lower() else (
            222, 207, 174, 255)
        packed_color = color[0] | color[1] << 8 | color[2] << 16 | color[3] << 24
        uv_data = mesh.uv_layers.active.data if mesh.uv_layers.active else None
        for triangle in mesh.loop_triangles:
            points = [obj.matrix_world @ mesh.vertices[i].co
                      for i in triangle.vertices]
            uvs = ([tuple(uv_data[i].uv) for i in triangle.loops]
                   if uv_data else [(0.0, 0.0)] * 3)
            records.append((points, uvs, packed_color))
        evaluated.to_mesh_clear()

    # Center and uniformly scale the whole four-piece set. Blender is Z-up;
    # the CUDA lab is Y-up.
    points = [point for triangle, _, _ in records for point in triangle]
    lo = Vector((min(p.x for p in points), min(p.y for p in points),
                 min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points),
                 max(p.z for p in points)))
    center = (lo + hi) * 0.5
    scale = 2.4 / max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)

    with open(args.output, "wb") as output:
        output.write(struct.pack("<8sI", b"BRTRI003", len(records)))
        for triangle, uvs, color in records:
            vertices = []
            for point in triangle:
                p = (point - center) * scale
                vertices.append(Vector((p.x, p.z, -p.y)))
            # Triangle stores one absolute vertex followed by two edge vectors:
            # P(u,v) = A + u*(B-A) + v*(C-A).
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
        principled = next((node for node in material.node_tree.nodes
                           if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        socket = principled.inputs.get("Base Color")
        if socket and socket.is_linked:
            source = socket.links[0].from_node
            if source.type == "TEX_IMAGE":
                base_color = source.image
                break
    if base_color is None:
        base_color = next((image for image in bpy.data.images
                           if "basecolor" in image.name.lower()), None)
    if base_color is None:
        raise RuntimeError("No embedded base-color image found")
    base_color.filepath_raw = args.texture_output
    base_color.file_format = "PNG"
    base_color.save()
    print(f"{args.input}: {len(records)} triangles -> {args.output}")


if __name__ == "__main__":
    main()

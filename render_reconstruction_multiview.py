#!/usr/bin/env python3
"""为 PGSR 恢复后的网格生成多角度白模截图。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="渲染恢复网格的多角度截图")
    parser.add_argument("--mesh", type=Path, default=root / "output/sanxingdui_mask/mesh/tsdf_fusion_post.ply")
    parser.add_argument("--output", type=Path, default=root / "output/sanxingdui_mask/mesh/multiview_36")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--azimuth-step", type=int, default=30, help="水平角间隔（度）")
    parser.add_argument("--elevations", type=float, nargs="+", default=[-25, 0, 25], help="俯仰角列表（度）")
    parser.add_argument("--vertical-fov", type=float, default=46.0, help="相机垂直视场角（度）")
    parser.add_argument("--distance-scale", type=float, default=1.12, help="相机距离安全系数")
    args = parser.parse_args()

    if args.azimuth_step <= 0 or 360 % args.azimuth_step != 0:
        raise ValueError("--azimuth-step 必须是能够整除 360 的正整数")
    if not args.mesh.is_file():
        raise FileNotFoundError(f"找不到输入网格：{args.mesh}")
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty():
        raise RuntimeError(f"无法读取网格：{args.mesh}")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.90, 0.90, 0.90])

    box = mesh.get_axis_aligned_bounding_box()
    center = box.get_center()
    radius = float(np.linalg.norm(box.get_extent()) / 2.0)
    distance = radius / math.tan(math.radians(args.vertical_fov / 2.0)) * args.distance_scale

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([0.035, 0.035, 0.035, 1.0])
    renderer.scene.scene.set_sun_light(np.array([-0.4, -0.5, -1.0], dtype=np.float32), np.ones(3, dtype=np.float32), 68000.0)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.enable_indirect_light(True)
    renderer.scene.scene.set_indirect_light_intensity(26000.0)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.90, 0.90, 0.90, 1.0]
    material.base_roughness = 0.82
    material.base_reflectance = 0.22
    renderer.scene.add_geometry("reconstructed_mask", mesh, material)

    rows = []
    index = 1
    for elevation_deg in args.elevations:
        elevation = math.radians(elevation_deg)
        for azimuth_deg in range(0, 360, args.azimuth_step):
            azimuth = math.radians(azimuth_deg)
            direction = np.array([
                math.sin(azimuth) * math.cos(elevation),
                math.sin(elevation),
                math.cos(azimuth) * math.cos(elevation),
            ], dtype=np.float32)
            eye = center + direction * distance
            renderer.setup_camera(args.vertical_fov, center, eye, np.array([0.0, 1.0, 0.0]))
            image = renderer.render_to_image()
            elevation_label = (
                f"{elevation_deg:+03.0f}"
                if float(elevation_deg).is_integer()
                else f"{elevation_deg:+05.1f}"
            )
            name = f"恢复视图_{index:02d}_方位{azimuth_deg:03d}_俯仰{elevation_label}.png"
            o3d.io.write_image(str(args.output / name), image, 9)
            rows.append((name, azimuth_deg, elevation_deg, *eye.tolist()))
            index += 1

    with (args.output / "相机角度.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["文件名", "方位角（度）", "俯仰角（度）", "相机X", "相机Y", "相机Z"])
        writer.writerows(rows)

    print(f"已生成 {len(rows)} 张截图：{args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""用已知相机和 Alpha 轮廓剔除 PGSR 网格中的多视图不一致表面。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="PGSR TSDF 网格的多视图轮廓雕刻")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=7, help="轮廓容差（像素）")
    parser.add_argument("--max-outside-ratio", type=float, default=0.03,
                        help="顶点最多允许落在多少比例的轮廓之外")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty():
        raise RuntimeError(f"无法读取网格：{args.mesh}")
    vertices = np.asarray(mesh.vertices)
    cameras = json.loads(args.cameras.read_text(encoding="utf-8"))
    outside = np.zeros(len(vertices), dtype=np.uint16)
    kernel = np.ones((args.dilate, args.dilate), np.uint8)

    for camera in cameras:
        image_path = args.images / f"{camera['img_name']}.png"
        alpha = np.asarray(Image.open(image_path).getchannel("A"))
        alpha = cv2.dilate((alpha > 127).astype(np.uint8), kernel)
        position = np.asarray(camera["position"], dtype=np.float64)
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        camera_points = (vertices - position) @ rotation
        z = camera_points[:, 2]
        u = camera_points[:, 0] * camera["fx"] / z + camera["width"] / 2
        v = camera_points[:, 1] * camera["fy"] / z + camera["height"] / 2
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = (z > 0) & (ui >= 0) & (ui < camera["width"]) & (vi >= 0) & (vi < camera["height"])
        inside = np.zeros(len(vertices), dtype=bool)
        inside[valid] = alpha[vi[valid], ui[valid]] > 0
        outside += ~inside

    limit = max(1, int(np.floor(len(cameras) * args.max_outside_ratio)))
    print("顶点轮廓外次数分位数：", np.percentile(outside, [0, 10, 25, 50, 75, 90, 100]))
    remove = outside > limit
    print(f"轮廓外次数允许值：{limit}/{len(cameras)}")
    print(f"删除顶点：{remove.sum()}/{len(remove)} ({remove.mean() * 100:.2f}%)")
    mesh.remove_vertices_by_mask(remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    # 雕刻后只保留三角形数量最多的连通部分。
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh):
        raise RuntimeError(f"写入失败：{args.output}")
    print(f"输出：{args.output}，{len(mesh.vertices)} 顶点，{len(mesh.triangles)} 三角形")


if __name__ == "__main__":
    main()

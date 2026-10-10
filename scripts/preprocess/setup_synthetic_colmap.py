#!/usr/bin/env python3
"""Create a COLMAP text model from the known Open3D render cameras."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return -qvec if qvec[0] < 0 else qvec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=Path, required=True, help="多视角图片所在目录")
    parser.add_argument("--poses", type=Path, required=True, help="渲染脚本生成的 camera_poses.csv")
    parser.add_argument("--output", type=Path, required=True, help="PGSR 数据集输出目录")
    parser.add_argument("--image-pattern", default="view_*.png", help="输入图片匹配表达式")
    parser.add_argument("--vertical-fov", type=float, default=48.0)
    parser.add_argument("--points", type=int, default=30000)
    parser.add_argument("--seed-radius-ratio", type=float, default=0.42)
    args = parser.parse_args()

    if not args.views.is_dir():
        raise FileNotFoundError(f"找不到视图目录：{args.views}")
    if not args.poses.is_file():
        raise FileNotFoundError(f"找不到相机位姿文件：{args.poses}")
    image_paths = sorted(args.views.glob(args.image_pattern))
    if not image_paths:
        raise RuntimeError(f"在 {args.views} 中没有找到匹配 {args.image_pattern!r} 的图片")
    with Image.open(image_paths[0]) as image:
        width, height = image.size
    focal = (height / 2.0) / math.tan(math.radians(args.vertical_fov / 2.0))

    images_dir = args.output / "images"
    sparse_dir = args.output / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for source in image_paths:
        shutil.copy2(source, images_dir / source.name)

    with args.poses.open(newline="", encoding="utf-8") as handle:
        poses = {row["file"]: row for row in csv.DictReader(handle)}
    missing_poses = [path.name for path in image_paths if path.name not in poses]
    if missing_poses:
        raise RuntimeError(f"以下图片缺少相机位姿：{', '.join(missing_poses[:5])}")

    (sparse_dir / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {focal:.12f} {focal:.12f} {width / 2:.12f} {height / 2:.12f}\n",
        encoding="utf-8",
    )

    image_lines = []
    eyes = []
    targets = []
    for image_id, image_path in enumerate(image_paths, start=1):
        pose = poses[image_path.name]
        eye = np.array([float(pose[f"eye_{axis}"]) for axis in "xyz"])
        target = np.array([float(pose[f"target_{axis}"]) for axis in "xyz"])
        eyes.append(eye)
        targets.append(target)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        rotation = np.stack([right, -camera_up, forward])
        translation = -rotation @ eye
        quaternion = rotmat2qvec(rotation)
        values = [*quaternion.tolist(), *translation.tolist()]
        image_lines.append(
            f"{image_id} " + " ".join(f"{value:.12f}" for value in values) + f" 1 {image_path.name}\n\n"
        )
    (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")

    # Use only camera metadata to initialize a random cloud. Sampling the source
    # OBJ here would leak the ground-truth geometry into a nominally 2D-only run.
    rng = np.random.default_rng(2026)
    target_center = np.mean(np.stack(targets), axis=0)
    camera_distance = np.median(np.linalg.norm(np.stack(eyes) - target_center, axis=1))
    seed_radius = camera_distance * args.seed_radius_ratio
    directions = rng.normal(size=(args.points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = seed_radius * np.cbrt(rng.random((args.points, 1)))
    vertices = target_center + directions * radii
    point_lines = [
        f"{point_id} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} 220 220 220 0.1\n"
        for point_id, xyz in enumerate(vertices, start=1)
    ]
    (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")
    print(f"Prepared {len(image_paths)} cameras and {len(vertices)} seed points in {args.output}")


if __name__ == "__main__":
    main()

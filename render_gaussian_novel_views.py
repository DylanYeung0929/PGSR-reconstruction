#!/usr/bin/env python3
"""从训练中未出现的中间相机角度渲染 PGSR 高斯模型。"""

from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import Camera
from utils.general_utils import safe_state


def make_camera(template: Camera, center: np.ndarray, distance: float,
                azimuth_deg: float, elevation_deg: float, uid: int) -> Camera:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    direction = np.array([
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    ])
    eye = center + direction * distance
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    world_to_camera = np.stack([right, -camera_up, forward])
    translation = -world_to_camera @ eye

    return Camera(
        colmap_id=100000 + uid,
        R=world_to_camera.T,
        T=translation,
        FoVx=template.FoVx,
        FoVy=template.FoVy,
        image_width=template.image_width,
        image_height=template.image_height,
        image_path=template.image_path,
        image_name=f"novel_{uid:02d}",
        uid=100000 + uid,
        trans=template.trans,
        scale=template.scale,
        preload_img=False,
        data_device="cuda",
    )


def main() -> None:
    parser = ArgumentParser(description="PGSR 高斯新视角渲染")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--novel_output", default="novel_gaussian_views", type=str,
                        help="输出目录；相对路径将放在模型目录下，也可传绝对路径")
    parser.add_argument("--azimuth-start", default=15, type=int, help="第一个水平角（度）")
    parser.add_argument("--azimuth-step", default=30, type=int, help="水平角间隔（度）")
    parser.add_argument("--elevations", default=[-12.5, 12.5], type=float, nargs="+", help="俯仰角列表（度）")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.azimuth_step <= 0:
        parser.error("--azimuth-step 必须是正整数")
    safe_state(args.quiet)

    dataset = model_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    pipeline = pipeline_params.extract(args)
    cameras = scene.getTrainCameras()
    template = cameras[0]
    centers = np.stack([camera.camera_center.detach().cpu().numpy() for camera in cameras])
    center = centers.mean(axis=0)
    distance = float(np.linalg.norm(centers - center, axis=1).mean())

    requested_output = Path(args.novel_output)
    output = requested_output if requested_output.is_absolute() else Path(dataset.model_path) / requested_output
    output.mkdir(parents=True, exist_ok=True)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    uid = 1
    with torch.no_grad():
        for elevation in args.elevations:
            for azimuth in range(args.azimuth_start, 360, args.azimuth_step):
                camera = make_camera(template, center, distance, azimuth, elevation, uid)
                image = render(camera, gaussians, pipeline, background)["render"].clamp(0.0, 1.0)
                name = f"高斯新视角_{uid:02d}_方位{azimuth:03d}_俯仰{elevation:+05.1f}.png"
                torchvision.utils.save_image(image, output / name)
                uid += 1
    print(f"已生成 {uid - 1} 张未训练角度的高斯渲染图：{output}")


if __name__ == "__main__":
    main()

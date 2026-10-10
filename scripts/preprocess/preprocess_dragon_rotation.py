#!/usr/bin/env python3
"""为转台旋转序列准备COLMAP/PGSR输入。

输入是固定相机拍摄、物体旋转的RGB视频帧。脚本会：
1. 按步长抽帧；2. 裁掉边缘和转台底部；3. 对绿色幕布抠图；
4. 生成黑色纯背景的RGBA图与同名的COLMAP特征掩膜。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def foreground_mask(bgr: np.ndarray, lower_y_cut: int) -> np.ndarray:
    """去掉本数据集的低饱和绿幕，保留龙的红/灰色像素。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 绿幕范围：H约70--95；因照明不均预留较宽范围。
    green = (
        (hsv[:, :, 0] >= 45)
        & (hsv[:, :, 0] <= 110)
        & (hsv[:, :, 1] >= 15)
        & (hsv[:, :, 2] >= 50)
    )
    mask = (~green).astype(np.uint8) * 255
    # 底部是固定转台；移除它，避免固定背景干扰转台SfM。
    mask[lower_y_cut:, :] = 0
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # 去除零星噪点，同时保留龙的翅膀、角等可能分离的部件。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 100:
            cleaned[labels == label] = 255
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--left", type=int, default=72)
    parser.add_argument("--right", type=int, default=568)
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--bottom", type=int, default=286)
    parser.add_argument("--base-cut", type=int, default=262,
                        help="裁切后图像中移除转台的起始Y坐标")
    args = parser.parse_args()

    images_out = args.output / "images"
    masks_out = args.output / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.input.glob("*.png"))
    selected = frames[::args.stride]
    if not selected:
        raise RuntimeError(f"未在 {args.input} 找到PNG图片")

    for output_index, frame_path in enumerate(selected):
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"无法读取：{frame_path}")
        crop = bgr[args.top:args.bottom, args.left:args.right]
        if crop.size == 0:
            raise RuntimeError("裁切范围无效")
        mask = foreground_mask(crop, args.base_cut)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # 背景颜色统一为黑色，防止背景纹理被训练为物体的一部分。
        rgb[mask == 0] = 0
        rgba = np.dstack((rgb, mask))
        name = f"frame_{output_index:04d}.png"
        cv2.imwrite(str(images_out / name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(masks_out / name), mask)

    print(f"已生成 {len(selected)} 张RGBA图片：{images_out}")
    print(f"COLMAP特征掩膜目录：{masks_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""渲染 PGSR 项目介绍用的中文思维导图 PNG。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def box(draw: ImageDraw.ImageDraw, rect, color, title, lines, number):
    x0, y0, x1, y1 = rect
    shadow = 9
    draw.rounded_rectangle((x0 + shadow, y0 + shadow, x1 + shadow, y1 + shadow), 28, fill="#02080d")
    draw.rounded_rectangle(rect, 28, fill="#102d40", outline=color, width=3)
    draw.ellipse((x0 + 25, y0 + 25, x0 + 77, y0 + 77), fill=color)
    draw.text((x0 + 44, y0 + 51), str(number), fill="#07131f", font=font(24), anchor="mm")
    draw.text((x0 + 95, y0 + 37), title, fill="white", font=font(25))
    y = y0 + 98
    for line, small in lines:
        draw.text((x0 + 42, y), line, fill="#d7e6f1" if not small else "#b8cad7", font=font(17 if not small else 15))
        y += 34 if not small else 40


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "output/project_intro/PGSR三维重建项目思维导图.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (WIDTH, HEIGHT), "#07131f")
    draw = ImageDraw.Draw(image)
    # 柔和背景渐变与同心线，提升投屏时的层次感。
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line((0, y, WIDTH, y), fill=(7 + int(9*t), 19 + int(22*t), 31 + int(27*t)))
    for radius in (370, 285):
        draw.ellipse((960-radius, 535-radius, 960+radius, 535+radius), outline="#17394b", width=1)

    draw.text((90, 70), "PGSR 多视图三维重建项目", fill="white", font=font(38))
    draw.text((92, 118), "从二维图像采集到高斯表达、TSDF 网格与可视化交付", fill="#c8d8e7", font=font(19))

    # 连线必须在节点之前绘制。
    lines = [
        ((805, 470), (495, 290), "#67c6e8"), ((803, 595), (485, 720), "#f4b658"),
        ((1118, 455), (1420, 275), "#a987ec"), ((1123, 615), (1420, 770), "#57d0a6"),
    ]
    for start, end, color in lines:
        draw.line((start, end), fill=color, width=5)
        x, y = end
        draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline="white", width=2)

    # 中心主题。
    draw.rounded_rectangle((815, 435, 1135, 655), 42, fill="#02080d")
    draw.rounded_rectangle((805, 425, 1125, 645), 42, fill="#268cc6", outline="#87e9ff", width=2)
    draw.text((965, 505), "PGSR", fill="white", font=font(40), anchor="mm")
    draw.text((965, 548), "二维视图 → 三维白模", fill="#e9fbff", font=font(20), anchor="mm")
    draw.text((965, 584), "Gaussian Splatting + TSDF", fill="#e9fbff", font=font(19), anchor="mm")

    box(draw, (115, 175, 495, 425), "#67c6e8", "数据采集", [
        ("• 虚拟相机：围绕 CAD / OBJ 白模截图", False),
        ("• 真实相机：环绕物体、相邻视角重叠", False),
        ("• 转台序列：抠出前景，排除固定背景", False),
        ("输入：RGB 图像序列（建议均匀覆盖 360°）", True),
    ], 1)
    box(draw, (105, 610, 485, 862), "#f4b658", "图像与位姿准备", [
        ("• 裁切、统一尺寸、白模 / 前景掩膜", False),
        ("• COLMAP：特征匹配、建图、位姿估计", False),
        ("• 输出 sparse 相机参数与稀疏点云", False),
        ("关键：无预设位姿时由 COLMAP 自动恢复", True),
    ], 2)
    box(draw, (1420, 155, 1805, 420), "#a987ec", "PGSR 训练", [
        ("• 以 COLMAP 点云初始化 3D 高斯", False),
        ("• 多视图光度、几何与轮廓约束", False),
        ("• 高斯增密、剪枝与迭代优化", False),
        ("产物：训练好的 Gaussian PLY 与深度渲染", True),
    ], 3)
    box(draw, (1420, 655, 1805, 920), "#57d0a6", "网格与交付", [
        ("• 渲染多视图深度 → TSDF 融合", False),
        ("• 连通域清理，输出三角网格 PLY", False),
        ("• 36 / 108 角度白模图与循环 GIF", False),
        ("交付：网格、训练高斯、数据集、运行脚本", True),
    ], 4)

    draw.text((960, 976), "已验证对象：三星堆面具 · 铜质加工件 · 石头加工件 · 龙转台序列", fill="#c8d8e7", font=font(19), anchor="mm")
    draw.text((960, 1020), "开源基线：zju3dv/PGSR ｜ 位姿估计：COLMAP ｜ 网格提取：TSDF Fusion", fill="#b8cad7", font=font(16), anchor="mm")
    image.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()

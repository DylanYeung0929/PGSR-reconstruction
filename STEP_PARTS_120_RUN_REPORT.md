# 两个 STEP 加工件的 PGSR 120视图复现报告

## 1. 输入文件

- `/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction/石头加工件.stp`
- `/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction/铜质加工件.stp`

STEP 只用于生成二维截图，不作为 PGSR 训练或 TSDF 融合的几何输入。

## 2. STEP 离散化

使用本项目新增脚本：

```bash
.venv/bin/python scripts/preprocess/step_to_mesh.py \
  --input ../石头加工件.stp \
  --output ../stone_part_reconstruction/source_mesh/stone_part.ply
```

铜件同理。脚本使用 OpenCascade 将 B样条曲面离散化，仅用于虚拟拍照。

- 石头件截图源网格：76,300顶点，139,703三角面；
- 铜质件截图源网格：112,067顶点，195,450三角面。

## 3. 120视图数据集

每个零件均使用：

- 24个水平方位角，间隔15°；
- 5个俯仰角：`-50°、-25°、0°、25°、50°`；
- 共 `24×5=120` 张；
- 分辨率 `1024×1024`；
- 白色材质、深灰背景、固定照明；
- 相机垂直视场角48°。

截图命令示例：

```bash
EGL_PLATFORM=surfaceless .venv/bin/python \
  ../sanxingdui-mask-model/render_white_multiview.py \
  --obj ../stone_part_reconstruction/source_mesh/stone_part.ply \
  --output ../stone_part_reconstruction/views_120 \
  --width 1024 --height 1024 \
  --azimuth-step 15 --elevations -50 -25 0 25 50 \
  --vertical-fov 48 --distance-scale 1.18
```

## 4. PGSR数据准备

```bash
.venv/bin/python scripts/preprocess/setup_synthetic_colmap.py \
  --views ../stone_part_reconstruction/views_120 \
  --poses ../stone_part_reconstruction/views_120/camera_poses.csv \
  --output data/stone_part_120 \
  --image-pattern 'view_*.png' \
  --vertical-fov 48 --points 30000 --seed-radius-ratio 0.42
```

训练输入只包含120张图片、相机内外参和30,000个随机种子点。随机点不是从 STEP 表面采样的。

## 5. PGSR训练

本次根据相机绝对坐标计算出最近相机距离约20～33，因此使用 `multi_view_max_dis=36`。每个相机最终选中两个15°相邻视角。若沿用三星堆的4.0，邻居数为0，PGSR多视图损失不会生效。

```bash
.venv/bin/python train.py \
  -s data/stone_part_120 \
  -m output/stone_part_120 \
  -r 2 --iterations 30000 \
  --multi_view_max_dis 36 \
  --multi_view_max_angle 35 \
  --max_abs_split_points 0 \
  --opacity_cull_threshold 0.05
```

结果：

| 模型 | 训练迭代 | 高斯数量 | 高斯PLY大小 |
|---|---:|---:|---:|
| 石头加工件 | 30,000 | 52,406 | 12,998,218字节 |
| 铜质加工件 | 30,000 | 38,041 | 9,435,698字节 |

石头件第30,000次训练集评估 L1 约为0.00699。

## 6. PGSR TSDF网格

使用 PGSR 仓库的 `render.py` 渲染平面深度并执行 Open3D TSDF 集成：

```bash
.venv/bin/python render.py \
  -m output/stone_part_120 --iteration 30000 \
  --max_depth 200 --voxel_size 0.15 \
  --crop_radius 0 --use_depth_filter
```

两个最终后处理网格：

| 模型 | 顶点 | 三角面 | 文件大小 |
|---|---:|---:|---:|
| 石头加工件 | 70,663 | 132,606 | 3,632,047字节 |
| 铜质加工件 | 62,469 | 112,503 | 3,149,470字节 |

白模、低纹理输入导致预测深度存在整体尺度和位置漂移。最终网格轮廓可辨，但几何精度低于高斯渲染；报告保留这一真实结果，没有使用 STEP 原网格或其他重建算法替换 PGSR 网格。

## 8. 铜件重影修正版 V2

旧版铜件网格在侧视图中出现三层重复结构。诊断表明，问题并非单纯的TSDF体素过大，而是细长白模占画面比例小、PGSR默认整图损失受大面积背景主导，导致不同相机的预测深度不一致。

V2仍使用PGSR训练与PGSR的TSDF流程，做了以下调整：

- 虚拟相机使用 `--adaptive-framing`，按各视角包围盒调整距离；
- PNG写入 `--alpha-mask` 前景轮廓；
- PGSR训练增加可选 `--foreground_weight 0.7`；
- `lambda_dssim` 从0.2调整到0.1；
- `densify_grad_threshold=0.001`，在10,000次停止增密，避免前景加权引起点数失控；
- 仍完整训练30,000次，并保留多视图几何与光度约束；
- TSDF使用0.10体素和Alpha轮廓，最终推荐不启用额外法线过滤的版本。

训练命令：

```bash
.venv/bin/python train.py \
  -s data/copper_part_120_v2 \
  -m output/copper_part_120_v2 -r 2 \
  --iterations 30000 \
  --multi_view_max_dis 36 --multi_view_max_angle 35 \
  --foreground_weight 0.7 --lambda_dssim 0.1 \
  --densify_grad_threshold 0.001 --densify_until_iter 10000 \
  --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

V2结果：

- 训练集 L1：约0.000922；
- PSNR：约45.83 dB；
- 高斯数量：23,303；
- 推荐网格：130,341顶点、249,222三角面；
- 旧版侧面的三重轮廓已消除，但白模低纹理造成的局部毛刺仍然存在。

推荐文件：

```text
output/copper_part_120_v2/point_cloud/iteration_30000/point_cloud.ply
output/copper_part_120_v2/mesh/tsdf_fusion_post.ply
output/copper_part_120_v2/mesh/multiview_36_no_depthfilter/
```

同时保留的法线过滤版本：

```text
output/copper_part_120_v2/mesh/tsdf_fusion_post_depthfilter.ply
output/copper_part_120_v2/mesh/multiview_36/
```

## 7. 结果目录

```text
../stone_part_reconstruction/
├── source_mesh/stone_part.ply
└── views_120/                         # 120张输入图、位姿CSV、总览图

../copper_part_reconstruction/
├── source_mesh/copper_part.ply
└── views_120/

output/stone_part_120/
├── point_cloud/iteration_30000/point_cloud.ply
├── novel_gaussian_views/              # 48张高斯新视角图及总览
└── mesh/
    ├── tsdf_fusion.ply
    ├── tsdf_fusion_post.ply
    └── multiview_36/                   # 36张最终网格图及总览

output/copper_part_120/                 # 目录结构相同
```

## 8. 关于方法来源

训练由该 GitHub 仓库的 `train.py`、`gaussian_renderer/`、`scene/` 和 CUDA 光栅化扩展完成；最终网格由同一仓库的 `render.py` 按其 PGSR 平面深度执行 TSDF 融合。OpenCascade只用于读取 STEP 并为二维数据采集生成截图网格，不参与重建。

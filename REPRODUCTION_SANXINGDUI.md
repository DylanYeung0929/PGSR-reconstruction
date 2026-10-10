# PGSR 从二维截图到三维模型的完整复现指南

本文说明如何从 OBJ/PLY 网格生成多视角二维白模图片，再仅使用图片和相机参数训练 PGSR，最后导出高斯模型、TSDF 网格和多角度结果图。示例路径均可替换，脚本已提供参数，无须修改 Python 源码。

> 源 OBJ 只在第一步生成二维图片时使用，训练不会读取它。当前流程属于“已知虚拟相机位姿的合成数据重建”。真实照片须先用 COLMAP 求相机参数，不能用第二步虚构位姿。

## 一、文件和目录

```text
/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction/
├── sanxingdui-mask-model/
│   ├── render_white_multiview.py          # 源网格→二维白模视图
│   ├── source/.../Sanxingdui Mask Model.obj
│   └── data/white_multiview_36/           # 图片及 camera_poses.csv
└── PGSR/
    ├── .venv/                             # Python环境
    ├── train.py                           # 训练
    ├── render.py                          # 深度渲染及TSDF融合
    ├── run_sanxingdui.sh                  # 可配置总控脚本
    ├── paths.example.env                  # 路径配置模板
    ├── render_gaussian_novel_views.py     # 高斯新视角截图
    ├── render_reconstruction_multiview.py # 最终网格截图
    └── scripts/preprocess/setup_synthetic_colmap.py
```

## 二、设置本次实验路径

变量仅在当前终端有效。新实验应使用新的 `VIEWS`、`DATASET`、`MODEL`，避免覆盖旧结果。

```bash
ROOT=/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction
PGSR_DIR="$ROOT/PGSR"
OBJ="$ROOT/sanxingdui-mask-model/source/Sanxingdui Mask Model/Sanxingdui Mask Model.obj"
VIEWS="$ROOT/sanxingdui-mask-model/data/white_multiview_36"
DATASET="$PGSR_DIR/data/sanxingdui_mask_manual"
MODEL="$PGSR_DIR/output/sanxingdui_mask_manual"
PYTHON="$PGSR_DIR/.venv/bin/python"

cd "$PGSR_DIR"
export CUDA_HOME=/usr/local/cuda-12.6
export EGL_PLATFORM=surfaceless
```

确认 GPU 环境：

```bash
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 三、步骤1：生成多视角二维白模图片

```bash
"$PYTHON" "$ROOT/sanxingdui-mask-model/render_white_multiview.py" \
  --obj "$OBJ" \
  --output "$VIEWS" \
  --width 1024 --height 1024 \
  --azimuth-step 30 \
  --elevations -25 0 25 \
  --vertical-fov 48 \
  --distance-scale 1.12
```

主要参数：

- `--obj`：输入 OBJ/PLY 网格；
- `--output`：图片输出目录；
- `--azimuth-step 30`：水平每隔30°截图；
- `--elevations`：俯仰角列表；
- `--vertical-fov`：垂直视场角；
- `--distance-scale`：相机距离安全系数，模型被裁切时适当增大。

本例生成 `12×3=36` 张 PNG 和 `camera_poses.csv`。检查：

```bash
find "$VIEWS" -maxdepth 1 -name 'view_*.png' | wc -l
head -n 3 "$VIEWS/camera_poses.csv"
```

若要72张，可改成 `--azimuth-step 15` 并保持三个俯仰层。

## 四、步骤2：转换为 PGSR/COLMAP 数据格式

```bash
cd "$PGSR_DIR"
"$PYTHON" scripts/preprocess/setup_synthetic_colmap.py \
  --views "$VIEWS" \
  --poses "$VIEWS/camera_poses.csv" \
  --output "$DATASET" \
  --image-pattern 'view_*.png' \
  --vertical-fov 48 \
  --points 30000 \
  --seed-radius-ratio 0.42
```

这里的 `--vertical-fov` 必须与步骤1相同，否则焦距错误。脚本不会读取 OBJ，而是根据相机覆盖范围随机生成初始点，避免把原几何直接交给重建算法。

```text
$DATASET/
├── images/                 # 训练图片
└── sparse/
    ├── cameras.txt         # 图像尺寸、fx、fy、cx、cy
    ├── images.txt          # 图片名、相机四元数及位移
    └── points3D.txt        # 随机初始稀疏点
```

本项目读取 `sparse/`，不是 `sparse/0/`。首次加载后会生成 `points3D.ply`。检查：

```bash
find "$DATASET/images" -maxdepth 1 -type f | wc -l
head -n 2 "$DATASET/sparse/cameras.txt"
head -n 4 "$DATASET/sparse/images.txt"
head -n 2 "$DATASET/sparse/points3D.txt"
```

## 五、步骤3：5次迭代环境测试

```bash
SMOKE="$PGSR_DIR/output/smoke_manual_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" train.py \
  -s "$DATASET" -m "$SMOKE" -r 2 \
  --iterations 5 --test_iterations 5 --save_iterations 5 \
  --multi_view_max_dis 4.0 --multi_view_max_angle 35 \
  --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

出现 `Training complete` 表示数据、CUDA和扩展基本可用。`-r 2` 将边长缩小一半，以适配8GB显存。

## 六、步骤4：从头训练高斯模型

确保 `$MODEL` 是新的输出目录：

```bash
"$PYTHON" train.py \
  -s "$DATASET" \
  -m "$MODEL" \
  -r 2 \
  --iterations 30000 \
  --multi_view_max_dis 4.0 \
  --multi_view_max_angle 35 \
  --max_abs_split_points 0 \
  --opacity_cull_threshold 0.05
```

`-s` 是数据集，`-m` 是训练输出。当前视角距离较大，因此使用 `multi_view_max_dis=4.0`。核心输出是：

```text
$MODEL/point_cloud/iteration_30000/point_cloud.ply
```

该 PLY 是包含位置、尺度、旋转、透明度和球谐颜色系数的三维高斯集合，不是普通三角网格。

## 七、步骤5：从新角度渲染高斯

```bash
"$PYTHON" render_gaussian_novel_views.py \
  -m "$MODEL" \
  --iteration 30000 \
  --novel_output "$MODEL/novel_gaussian_views" \
  --azimuth-start 15 --azimuth-step 30 \
  --elevations -12.5 12.5
```

`--novel_output` 可传绝对路径；相对路径则位于模型目录下。脚本通过 `$MODEL/cfg_args` 自动读取原数据集路径。

## 八、步骤6：TSDF融合并导出最终网格

```bash
"$PYTHON" render.py \
  -m "$MODEL" \
  --iteration 30000 \
  --max_depth 15.0 \
  --voxel_size 0.03 \
  --crop_radius 2.7 \
  --crop_y_radius 1.75 \
  --use_depth_filter
```

- `max_depth`：忽略过远深度；
- `voxel_size`：TSDF体素边长，越小越密，但开销和噪声可能增加；
- `crop_radius`、`crop_y_radius`：清除主体以外浮层；
- `use_depth_filter`：过滤不可靠深度。

核心输出：

```text
$MODEL/mesh/tsdf_fusion.ply
$MODEL/mesh/tsdf_fusion_post.ply
```

通常使用清理后的 `tsdf_fusion_post.ply`。高斯适合照片级新视角渲染；网格适合 Blender、CAD、切片、碰撞和常规三维处理。

## 九、步骤7：生成最终网格多角度截图

```bash
MESH="$MODEL/mesh/tsdf_fusion_post.ply"
MESH_VIEWS="$MODEL/mesh/multiview_36"

EGL_PLATFORM=surfaceless "$PYTHON" render_reconstruction_multiview.py \
  --mesh "$MESH" \
  --output "$MESH_VIEWS" \
  --width 1024 --height 1024 \
  --azimuth-step 30 \
  --elevations -25 0 25 \
  --vertical-fov 46 \
  --distance-scale 1.12
```

输出包含36张白模截图和 `相机角度.csv`。更换 `--mesh` 和 `--output` 即可处理其他实验。

## 十、用配置文件控制一键脚本

复制模板并修改其中路径：

```bash
cd "$PGSR_DIR"
cp paths.example.env paths.my_experiment.env
```

然后执行：

```bash
CONFIG_FILE="$PGSR_DIR/paths.my_experiment.env" \
RUN_NAME=my_experiment_01 \
./run_sanxingdui.sh fresh
```

也可直接临时传入路径：

```bash
VIEWS=/新图片目录 \
POSES=/新图片目录/camera_poses.csv \
DATASET=/新数据集目录 \
RUNS_ROOT=/新结果根目录 \
RUN_NAME=experiment_02 \
./run_sanxingdui.sh fresh
```

模式说明：

```text
prepare      只转换数据
train        训练（已有 iteration_30000 时跳过）
mesh         只提取网格
smoke        5次迭代测试
all          默认目录执行全部阶段
fresh        新目录从头训练并提取网格
fresh-train  新目录只训练高斯
```

## 十一、换模型时的检查清单

1. 更换 `OBJ`、`VIEWS`、`DATASET`、`MODEL` 四个路径。
2. 步骤1和步骤2的 `vertical-fov` 完全相同。
3. 图片与 `camera_poses.csv` 来自同一次渲染。
4. 图片名能被 `--image-pattern` 匹配，并在CSV中有同名记录。
5. 不把新数据或结果写入旧实验目录。
6. 正式训练前先跑5次迭代测试。

## 十二、常见问题

- Open3D 报 EGL/无显示器错误：设置 `EGL_PLATFORM=surfaceless`。
- CUDA不可用：运行 `nvidia-smi`，并检查 `/usr/local/cuda-12.6/bin/nvcc` 和 `torch.cuda.is_available()`。
- 训练瞬间结束：旧目录已有 `iteration_30000/point_cloud.ply`，使用新的 `MODEL` 或 `fresh`。
- 输入模型被裁切：增大步骤1的 `distance-scale`。
- 最终网格被裁切：增大 `crop_radius` 或 `crop_y_radius`。
- 网格比高斯模糊：高斯直接拟合图像，TSDF还需将深度离散到体素；增加视图、改善覆盖或适度减小体素可改善，但更小体素也可能放大噪声。
- 使用真实照片：不能运行合成位姿转换脚本，应先用 COLMAP生成相机内外参和稀疏点云，再向 PGSR提供 `images/` 与 `sparse/`。

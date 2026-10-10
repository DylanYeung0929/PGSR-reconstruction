#!/usr/bin/env bash
# 龙转台图像：COLMAP 自动估计位姿 + PGSR 训练 + TSDF 网格融合。
# 使用前只需按需改下面的路径、抽帧步长和 MODEL_ID；不会修改原始帧目录。
set -euo pipefail

PGSR_DIR="/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction/PGSR"
RAW_IMAGES="/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction/GS_rotation_dargon_001_rgb_png"
DATASET_DIR="$PGSR_DIR/data/dragon_rotation_132_colmap"
OUTPUT_DIR="$PGSR_DIR/output/dragon_rotation_132_colmap"
PYTHON="$PGSR_DIR/.venv/bin/python"
COLMAP_BIN="$(command -v colmap)"

# 原始序列394帧；stride=3 产生132帧。MODEL_ID 是 COLMAP Mapper 的完整模型编号。
STRIDE=3
MODEL_ID=1
ITERATIONS=30000
# 若要从头覆盖同名预处理数据，执行时显式加 CLEAN_DATASET=1。
CLEAN_DATASET="${CLEAN_DATASET:-0}"

cd "$PGSR_DIR"

# 1) 预处理：裁切、绿幕抠图，生成 PGSR 的 RGBA 图和 COLMAP 掩膜。
# 为防止误删已有实验数据，默认拒绝覆盖；确认需要重跑时才允许清理这一套派生数据。
if [[ -e "$DATASET_DIR/database.db" || -d "$DATASET_DIR/images" ]]; then
  if [[ "$CLEAN_DATASET" != "1" ]]; then
    echo "检测到已有预处理数据：$DATASET_DIR" >&2
    echo "如确认从 COLMAP 步骤重新开始，请执行：CLEAN_DATASET=1 $0" >&2
    exit 2
  fi
  rm -rf "$DATASET_DIR/images" "$DATASET_DIR/masks" "$DATASET_DIR/sparse_reconstruction" "$DATASET_DIR/sparse" "$DATASET_DIR/database.db"
fi
"$PYTHON" scripts/preprocess/preprocess_dragon_rotation.py \
  --input "$RAW_IMAGES" --output "$DATASET_DIR" --stride "$STRIDE"

# 2) 仅由图像特征估计相机内外参。没有传入任何外部相机位姿。
"$COLMAP_BIN" feature_extractor \
  --database_path "$DATASET_DIR/database.db" --image_path "$DATASET_DIR/images" \
  --ImageReader.mask_path "$DATASET_DIR/masks" --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1
"$COLMAP_BIN" exhaustive_matcher --database_path "$DATASET_DIR/database.db" \
  --SiftMatching.use_gpu 0 --SiftMatching.guided_matching 1
"$COLMAP_BIN" mapper --database_path "$DATASET_DIR/database.db" \
  --image_path "$DATASET_DIR/images" --output_path "$DATASET_DIR/sparse_reconstruction" \
  --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_focal_length 1 --Mapper.min_num_matches 12

# PGSR 读取 sparse/ 下直接存在的三个 bin 文件，而不是 sparse/0/ 结构。
mkdir -p "$DATASET_DIR/sparse"
for file in cameras.bin images.bin points3D.bin; do
  ln -sfn "../sparse_reconstruction/$MODEL_ID/$file" "$DATASET_DIR/sparse/$file"
done

# 3) PGSR 高斯训练。输出点云会保存在 $OUTPUT_DIR/point_cloud/iteration_$ITERATIONS/。
"$PYTHON" train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" -r 1 --iterations "$ITERATIONS" \
  --multi_view_max_dis 2.0 --multi_view_max_angle 25 \
  --foreground_weight 0.8 --silhouette_weight 0.2 --lambda_dssim 0.1 \
  --multi_view_ncc_weight 0.01 --densify_grad_threshold 0.001 \
  --densify_until_iter 10000 --max_abs_split_points 0 --opacity_cull_threshold 0.05

# 4) 从训练好的高斯估计深度，再以 TSDF 融合输出三角网格。
"$PYTHON" render.py -m "$OUTPUT_DIR" --iteration "$ITERATIONS" --skip_test \
  --max_depth 20 --voxel_size 0.005 --num_cluster 1 --use_depth_filter

# 5) 导出36个白模视角和一张 6×6 总览图。
MESH="$OUTPUT_DIR/mesh/tsdf_fusion_post.ply"
VIEWS="$OUTPUT_DIR/mesh/multiview_36"
"$PYTHON" render_reconstruction_multiview.py --mesh "$MESH" --output "$VIEWS" \
  --width 1024 --height 1024 --azimuth-step 30 --elevations -25 0 25
if command -v gm >/dev/null; then
  gm montage -tile 6x6 -geometry 256x256+2+2 -background black \
    "$VIEWS"/*.png "$OUTPUT_DIR/mesh/龙_COLMAP_PGSR_36角度_6x6.jpg"
else
  montage -tile 6x6 -geometry 256x256+2+2 -background black \
    "$VIEWS"/*.png "$OUTPUT_DIR/mesh/龙_COLMAP_PGSR_36角度_6x6.jpg"
fi

echo "完成。最终网格：$MESH"
echo "36视图总览：$OUTPUT_DIR/mesh/龙_COLMAP_PGSR_36角度_6x6.jpg"

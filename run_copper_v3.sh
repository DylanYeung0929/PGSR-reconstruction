#!/usr/bin/env bash
set -euo pipefail

# ===== 常用路径：更换模型时优先修改这里 =====
ROOT="/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction"
PGSR="$ROOT/PGSR"
SOURCE_MESH="$ROOT/copper_part_reconstruction/source_mesh/copper_part.ply"
VIEWS="$ROOT/copper_part_reconstruction/views_216_v3_silhouette"
DATASET="$PGSR/data/copper_part_216_v3"
OUTPUT="$PGSR/output/copper_part_216_v3"
PYTHON="$PGSR/.venv/bin/python"

# ===== 数据集参数 =====
AZIMUTH_STEP=10
ELEVATIONS=(-60 -36 -12 12 36 60) # 36×6=216张
RESOLUTION=2                       # PGSR读取时缩小2倍

# ===== 训练参数 =====
ITERATIONS=30000
FOREGROUND_WEIGHT=0.7
SILHOUETTE_WEIGHT=0.1
MULTIVIEW_GEO_WEIGHT=0.03
MULTIVIEW_NCC_WEIGHT=0.01
DENSIFY_GRAD_THRESHOLD=0.001
DENSIFY_UNTIL_ITER=10000
OPACITY_CULL_THRESHOLD=0.05

# ===== TSDF参数 =====
VOXEL_SIZE=0.10
MAX_DEPTH=200

cd "$PGSR"

case "${1:-all}" in
  capture)
    python "$ROOT/sanxingdui-mask-model/render_white_multiview.py" \
      --obj "$SOURCE_MESH" --output "$VIEWS" --width 1024 --height 1024 \
      --azimuth-step "$AZIMUTH_STEP" --elevations "${ELEVATIONS[@]}" \
      --vertical-fov 48 --distance-scale 1.12 --adaptive-framing --alpha-mask
    ;;
  prepare)
    "$PYTHON" scripts/preprocess/setup_synthetic_colmap.py \
      --views "$VIEWS" --poses "$VIEWS/camera_poses.csv" --output "$DATASET" \
      --vertical-fov 48 --points 30000 --seed-radius-ratio 0.65
    ;;
  train)
    "$PYTHON" train.py -s "$DATASET" -m "$OUTPUT" -r "$RESOLUTION" \
      --iterations "$ITERATIONS" --multi_view_max_dis 30 --multi_view_max_angle 25 \
      --foreground_weight "$FOREGROUND_WEIGHT" --silhouette_weight "$SILHOUETTE_WEIGHT" \
      --multi_view_geo_weight "$MULTIVIEW_GEO_WEIGHT" --multi_view_ncc_weight "$MULTIVIEW_NCC_WEIGHT" \
      --lambda_dssim 0.1 --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
      --densify_until_iter "$DENSIFY_UNTIL_ITER" --max_abs_split_points 0 \
      --opacity_cull_threshold "$OPACITY_CULL_THRESHOLD"
    ;;
  mesh)
    "$PYTHON" render.py -m "$OUTPUT" --iteration "$ITERATIONS" --skip_test \
      --max_depth "$MAX_DEPTH" --voxel_size "$VOXEL_SIZE" --num_cluster 1 --use_depth_filter
    ;;
  screenshots)
    "$PYTHON" render_reconstruction_multiview.py \
      --mesh "$OUTPUT/mesh/tsdf_fusion_post.ply" \
      --output "$OUTPUT/mesh/multiview_36_final" --width 1024 --height 1024 \
      --azimuth-step 30 --elevations -25 0 25
    ;;
  all)
    "$0" capture
    "$0" prepare
    "$0" train
    "$0" mesh
    "$0" screenshots
    ;;
  *)
    echo "用法：$0 {capture|prepare|train|mesh|screenshots|all}" >&2
    exit 2
    ;;
esac

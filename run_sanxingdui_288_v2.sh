#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction"
PGSR="$ROOT/PGSR"
SOURCE="$ROOT/sanxingdui-mask-model/source/Sanxingdui Mask Model/Sanxingdui Mask Model.obj"
VIEWS="$ROOT/sanxingdui-mask-model/data/white_multiview_288_v2"
DATASET="$PGSR/data/sanxingdui_mask_288_v2"
OUTPUT="$PGSR/output/sanxingdui_mask_288_v2"
PYTHON="$PGSR/.venv/bin/python"

cd "$PGSR"
case "${1:-all}" in
  capture)
    python "$ROOT/sanxingdui-mask-model/render_white_multiview.py" \
      --obj "$SOURCE" --output "$VIEWS" --width 1024 --height 1024 \
      --azimuth-step 10 --elevations -70 -50 -30 -10 10 30 50 70 \
      --vertical-fov 48 --distance-scale 1.12 --adaptive-framing --alpha-mask ;;
  prepare)
    "$PYTHON" scripts/preprocess/setup_synthetic_colmap.py \
      --views "$VIEWS" --poses "$VIEWS/camera_poses.csv" --output "$DATASET" \
      --vertical-fov 48 --points 30000 --seed-radius-ratio 0.65 ;;
  train)
    "$PYTHON" train.py -s "$DATASET" -m "$OUTPUT" -r 2 --iterations 30000 \
      --multi_view_max_dis 2.0 --multi_view_max_angle 25 \
      --foreground_weight 0.7 --silhouette_weight 0.1 --lambda_dssim 0.1 \
      --multi_view_ncc_weight 0.01 --densify_grad_threshold 0.001 \
      --densify_until_iter 10000 --max_abs_split_points 0 --opacity_cull_threshold 0.05 ;;
  mesh)
    "$PYTHON" render.py -m "$OUTPUT" --iteration 30000 --skip_test \
      --max_depth 20 --voxel_size 0.005 --num_cluster 1 --use_depth_filter ;;
  screenshots)
    "$PYTHON" render_reconstruction_multiview.py \
      --mesh "$OUTPUT/mesh/tsdf_fusion_post.ply" \
      --output "$OUTPUT/mesh/multiview_36_final" --width 1024 --height 1024 \
      --azimuth-step 30 --elevations -25 0 25 ;;
  all)
    "$0" capture; "$0" prepare; "$0" train; "$0" mesh; "$0" screenshots ;;
  *) echo "用法：$0 {capture|prepare|train|mesh|screenshots|all}" >&2; exit 2 ;;
esac

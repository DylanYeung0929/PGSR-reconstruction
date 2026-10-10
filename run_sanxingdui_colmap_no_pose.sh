#!/usr/bin/env bash
set -euo pipefail

# 无已知相机位姿实验：只使用PNG截图，由COLMAP自动恢复相机与稀疏点云。
# 若要换模型/数据，只需修改下面四项路径。
ROOT="/home/dylanyeung/dylanyeung/study/Yue_Team/2D_to_3D_Reconstruction"
PGSR="$ROOT/PGSR"
VIEWS="$ROOT/sanxingdui-mask-model/data/white_multiview_288_v2"
DATASET="$PGSR/data/sanxingdui_colmap_no_pose"
OUTPUT="$PGSR/output/sanxingdui_colmap_no_pose"

PYTHON="$PGSR/.venv/bin/python"
COLMAP="${COLMAP:-$(command -v colmap)}"

cd "$PGSR"

prepare_images() {
  mkdir -p "$DATASET/sparse" "$DATASET/sparse_reconstruction"
  if [[ ! -e "$DATASET/images" ]]; then
    # 使用符号链接复用图片，避免重复占用约192 MB。
    ln -s "$VIEWS" "$DATASET/images"
  fi
}

run_colmap() {
  prepare_images
  if [[ ! -f "$DATASET/database.db" ]]; then
    "$COLMAP" feature_extractor \
      --database_path "$DATASET/database.db" \
      --image_path "$DATASET/images" \
      --ImageReader.single_camera 1 \
      --ImageReader.camera_model PINHOLE \
      --SiftExtraction.use_gpu 1

    "$COLMAP" exhaustive_matcher \
      --database_path "$DATASET/database.db" \
      --SiftMatching.use_gpu 0 \
      --SiftMatching.guided_matching 1
  fi

  if [[ ! -f "$DATASET/sparse_reconstruction/0/images.bin" ]]; then
    "$COLMAP" mapper \
      --database_path "$DATASET/database.db" \
      --image_path "$DATASET/images" \
      --output_path "$DATASET/sparse_reconstruction" \
      --Mapper.ba_refine_principal_point 0 \
      --Mapper.ba_refine_focal_length 1 \
      --Mapper.min_num_matches 12
  fi

  # 当前PGSR读取器要求三个bin文件直接位于 sparse/ 下。
  for file in cameras.bin images.bin points3D.bin; do
    [[ -e "$DATASET/sparse/$file" ]] || \
      ln -s "../sparse_reconstruction/0/$file" "$DATASET/sparse/$file"
  done
}

run_train() {
  run_colmap
  "$PYTHON" train.py \
    -s "$DATASET" -m "$OUTPUT" -r 2 --iterations 30000 \
    --multi_view_max_dis 2.0 --multi_view_max_angle 25 \
    --foreground_weight 0.7 --silhouette_weight 0.1 --lambda_dssim 0.1 \
    --multi_view_ncc_weight 0.01 --densify_grad_threshold 0.001 \
    --densify_until_iter 10000 --max_abs_split_points 0 \
    --opacity_cull_threshold 0.05
}

run_mesh() {
  "$PYTHON" render.py \
    -m "$OUTPUT" --iteration 30000 --skip_test \
    --max_depth 20 --voxel_size 0.005 --num_cluster 1 --use_depth_filter
}

run_screenshots() {
  "$PYTHON" render_reconstruction_multiview.py \
    --mesh "$OUTPUT/mesh/tsdf_fusion_post.ply" \
    --output "$OUTPUT/mesh/multiview_36" \
    --width 1024 --height 1024 --azimuth-step 30 --elevations -25 0 25
}

case "${1:-all}" in
  prepare) prepare_images ;;
  colmap) run_colmap ;;
  train) run_train ;;
  mesh) run_mesh ;;
  screenshots) run_screenshots ;;
  all) run_train; run_mesh; run_screenshots ;;
  *) echo "用法：$0 {prepare|colmap|train|mesh|screenshots|all}" >&2; exit 2 ;;
esac

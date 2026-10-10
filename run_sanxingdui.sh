#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"
# 可选：CONFIG_FILE=/绝对路径/paths.env ./run_sanxingdui.sh fresh
CONFIG_FILE="${CONFIG_FILE:-}"
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: config file not found: $CONFIG_FILE" >&2
        exit 1
    fi
    # 配置文件由用户维护，内容为 KEY=VALUE 格式。
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda-12.6}}"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
VIEWS="${VIEWS:-$SCRIPT_DIR/../sanxingdui-mask-model/data/white_multiview_36}"
POSES="${POSES:-$VIEWS/camera_poses.csv}"
DATASET="${DATASET:-$SCRIPT_DIR/data/sanxingdui_mask}"
SAVED_MODEL="${SAVED_MODEL:-$SCRIPT_DIR/output/sanxingdui_mask}"
RUNS_ROOT="${RUNS_ROOT:-$SCRIPT_DIR/output/sanxingdui_mask_runs}"
SMOKE_MODEL="${SMOKE_MODEL:-$SCRIPT_DIR/output/sanxingdui_smoke}"

# fresh / fresh-train 总是使用一个新目录，避免覆盖原有 PLY 和网格。
if [[ "$MODE" == "fresh" || "$MODE" == "fresh-train" ]]; then
    RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
    MODEL="$RUNS_ROOT/$RUN_NAME"
    if [[ -e "$MODEL" ]]; then
        echo "Error: new training output directory already exists: $MODEL" >&2
        echo "       Choose another name, for example: RUN_NAME=my_run_02 $0 $MODE" >&2
        exit 1
    fi
else
    MODEL="${MODEL_DIR:-$SAVED_MODEL}"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: PGSR environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$CUDA_ROOT" ]]; then
    echo "Error: CUDA toolkit not found: $CUDA_ROOT" >&2
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "Error: NVIDIA GPU/driver is unavailable." >&2
    exit 1
fi

export CUDA_HOME="$CUDA_ROOT"

prepare_data() {
    echo "[1/3] Preparing 36-view camera dataset..."
    "$PYTHON" scripts/preprocess/setup_synthetic_colmap.py \
        --views "$VIEWS" \
        --poses "$POSES" \
        --output "$DATASET"
}

train_model() {
    local final_model="$MODEL/point_cloud/iteration_30000/point_cloud.ply"
    if [[ -f "$final_model" ]]; then
        echo "[2/3] Trained model already exists; skipping training."
        echo "      To train from scratch without overwriting it, run: $0 fresh"
        return
    fi
    echo "      This run's output directory: $MODEL"
    echo "[2/3] Training PGSR for 30,000 iterations..."
    "$PYTHON" train.py \
        -s "$DATASET" \
        -m "$MODEL" \
        -r 2 \
        --multi_view_max_dis 4.0 \
        --multi_view_max_angle 35 \
        --max_abs_split_points 0 \
        --opacity_cull_threshold 0.05
}

extract_mesh() {
    if [[ ! -f "$MODEL/point_cloud/iteration_30000/point_cloud.ply" ]]; then
        echo "Error: the 30,000-iteration model is missing; run training first." >&2
        exit 1
    fi
    echo "[3/3] Rendering depth/normal maps and running TSDF fusion..."
    "$PYTHON" render.py \
        -m "$MODEL" \
        --max_depth 15.0 \
        --voxel_size 0.03 \
        --crop_radius 2.7 \
        --crop_y_radius 1.75 \
        --use_depth_filter
    echo "Final mesh: $MODEL/mesh/tsdf_fusion_post.ply"
}

smoke_test() {
    echo "Running a five-iteration GPU smoke test..."
    "$PYTHON" train.py \
        -s "$DATASET" \
        -m "$SMOKE_MODEL" \
        -r 2 \
        --iterations 5 \
        --test_iterations 5 \
        --save_iterations 5 \
        --multi_view_max_dis 4.0 \
        --multi_view_max_angle 35 \
        --max_abs_split_points 0 \
        --opacity_cull_threshold 0.05
}

case "$MODE" in
    all)
        prepare_data
        train_model
        extract_mesh
        ;;
    fresh)
        prepare_data
        train_model
        extract_mesh
        ;;
    fresh-train)
        prepare_data
        train_model
        echo "Training complete. Gaussian model: $MODEL/point_cloud/iteration_30000/point_cloud.ply"
        ;;
    prepare) prepare_data ;;
    train) train_model ;;
    mesh) extract_mesh ;;
    smoke) prepare_data; smoke_test ;;
    *)
        echo "Usage: $0 [all|fresh|fresh-train|prepare|train|mesh|smoke]" >&2
        exit 2
        ;;
esac

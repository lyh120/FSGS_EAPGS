#!/bin/bash
set -e

# ================= 根目录 =================
DATA_ROOT="/home/liuyuhao/FSGS/dataset"
COLMAP_ROOT="/home/liuyuhao/FSGS/colmap"

# 场景列表
SCENES=(
    # Chocolate
    # Laboratory
    # GearWorks
    Cupcake
    # Popcorn
    # Ujikintoki
    #MilkCookie
)

# 是否跑密集重建（3DGS建议 false）
DO_DENSE=false

# ================= 检查 colmap =================
if ! command -v colmap &> /dev/null; then
    echo "错误：未找到 colmap，请先安装"
    exit 1
fi

# ================= 主循环 =================
for SCENE in "${SCENES[@]}"; do
    echo "======================================"
    echo "开始处理场景: $SCENE"
    echo "======================================"

    IMAGE_DIR="$DATA_ROOT/$SCENE/train"
    WORKSPACE_DIR="$COLMAP_ROOT/$SCENE"

    DATABASE="$WORKSPACE_DIR/database.db"
    SPARSE_DIR="$WORKSPACE_DIR/sparse"
    DENSE_DIR="$WORKSPACE_DIR/dense"

    # 检查数据
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "⚠️ 跳过 $SCENE（未找到目录: $IMAGE_DIR）"
        continue
    fi

    # 创建目录
    mkdir -p "$WORKSPACE_DIR"
    mkdir -p "$SPARSE_DIR"

    echo "===== [$SCENE] 特征提取 ====="
    colmap feature_extractor \
        --database_path "$DATABASE" \
        --image_path "$IMAGE_DIR" \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 1

    echo "===== [$SCENE] 特征匹配 ====="
    colmap exhaustive_matcher \
        --database_path "$DATABASE" \
        --SiftMatching.use_gpu 1

    echo "===== [$SCENE] 稀疏重建 ====="
    colmap mapper \
        --database_path "$DATABASE" \
        --image_path "$IMAGE_DIR" \
        --output_path "$SPARSE_DIR"

    # ================= 密集（可选） =================
    if [ "$DO_DENSE" = true ]; then
        SPARSE_MODEL="$SPARSE_DIR/0"

        if [ -d "$SPARSE_MODEL" ]; then
            echo "===== [$SCENE] 去畸变 ====="
            colmap image_undistorter \
                --image_path "$IMAGE_DIR" \
                --input_path "$SPARSE_MODEL" \
                --output_path "$DENSE_DIR" \
                --output_type COLMAP

            echo "===== [$SCENE] PatchMatch ====="
            colmap patch_match_stereo \
                --workspace_path "$DENSE_DIR" \
                --workspace_format COLMAP \
                --PatchMatchStereo.geom_consistency true

            echo "===== [$SCENE] 点云融合 ====="
            colmap stereo_fusion \
                --workspace_path "$DENSE_DIR" \
                --workspace_format COLMAP \
                --input_type geometric \
                --output_path "$DENSE_DIR/fused.ply"
        else
            echo "⚠️ $SCENE 稀疏模型不存在，跳过 dense"
        fi
    fi

    echo "✅ 完成场景: $SCENE"
    echo "稀疏模型: $SPARSE_DIR/0"
    echo ""

done

echo "🎉 全部场景处理完成！"
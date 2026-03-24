#!/bin/bash

set -e

ITER=60000

TEST_ITERS="1000 3000 5000 10000 20000 30000 40000 50000 60000"
SAVE_ITERS="20000 40000 60000"
CKPT_ITERS="20000 40000 60000"

#Chocolate
    # Laboratory
    # GearWorks 
    # Cupcake
    # Popcorn
    # Ujikintoki
    #GearWorks
DATASETS=(
    Cupcake
)

for DATA in "${DATASETS[@]}"
do
    echo "=============================="
    echo "Training: $DATA"
    echo "=============================="

    # 👉 自动创建输出目录
    OUTDIR=output_v2_rest/${DATA,,}
    mkdir -p $OUTDIR

    # 👉 开始训练
    python train.py \
        --source_path dataset/$DATA \
        --model_path $OUTDIR \
        --n_views 0 \
        --eval \
        --resolution 1 \
        --iterations $ITER \
        --test_iterations $TEST_ITERS \
        --save_iterations $SAVE_ITERS \
        --checkpoint_iterations $CKPT_ITERS \
        \
        # ===== 核心优化参数（重点）=====
        --densify_until_iter 40000 \
        --densify_from_iter 500 \
        --densification_interval 100 \
        \
        --prune_from_iter 1000 \
        --prune_threshold 0.003 \
        \
        --opacity_reset_interval 3000 \
        \
        --position_lr_max_steps 60000 \
        \
        --depth_weight 0.1 \
        --depth_pseudo_weight 0.3 \
        \
        | tee logs_${DATA}.txt || echo "❌ Failed on $DATA, continuing..."

    echo "✅ Finished: $DATA"
done

echo "🎉 All training completed!"
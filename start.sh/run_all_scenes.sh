#!/bin/bash
# ============================================
# Batch run blender_to_colmap.py for 7 scenes
# ============================================

# SCENES=("Chocolate" "Laboratory" "GearWorks" "Cupcake" "Popcorn" "Ujikintoki" "MilkCookie")
SCENES=( "Cupcake" )

BASE_DATA_DIR="dataset"
BASE_COLMAP_DIR="colmap"
BASE_OUTPUT_DIR="colmap_unity"

PY_SCRIPT="blender_to_colmap.py"  # 如果不在当前目录，请使用绝对路径

for SCENE in "${SCENES[@]}"; do
    TRAIN_JSON="${BASE_DATA_DIR}/${SCENE}/transforms_train.json"
    TEST_JSON="${BASE_DATA_DIR}/${SCENE}/transforms_test.json"
    IMAGE_ROOT="${BASE_DATA_DIR}/${SCENE}"
    EXISTING_COLMAP="${BASE_COLMAP_DIR}/${SCENE}/sparse/0"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SCENE}/sparse/0"

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=== Processing scene: ${SCENE} ==="
    echo "Command: python ${PY_SCRIPT} --train-json ${TRAIN_JSON} --test-json ${TEST_JSON} --image-root ${IMAGE_ROOT} --existing-colmap ${EXISTING_COLMAP} --output-dir ${OUTPUT_DIR} --split-subdirs"

    python "${PY_SCRIPT}" \
        --train-json "${TRAIN_JSON}" \
        --test-json "${TEST_JSON}" \
        --image-root "${IMAGE_ROOT}" \
        --existing-colmap "${EXISTING_COLMAP}" \
        --output-dir "${OUTPUT_DIR}" \
        --split-subdirs

    if [ $? -ne 0 ]; then
        echo "⚠️  Scene ${SCENE} failed!"
    else
        echo "✅ Scene ${SCENE} done."
    fi
done

echo ""
echo "🎉 All scenes processed."
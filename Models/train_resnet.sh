#!/bin/bash
set -e
# --- Configuration ---
GAME_NAME="gallery"
BATCH_SIZE=4
LEARNING_RATE=5e-5
NUM_EPOCHS=14
BACKBONE="resnet50"

DIR_STEP1="Results/new/resnet_full"
DIR_ANALYSIS="Results/new/exploration_resnet"
DIR_STEP2="Results/new/resnet_ours"

TRAIN_SCRIPT="resnet_lstm.py"
EXPLORE_SCRIPT="exploration_resnet.py"

echo "========================================================"
echo "Pipeline Started for Game: ${GAME_NAME} (Backbone: ${BACKBONE})"
echo "========================================================"

echo ""
echo "[Step 1/3] Starting Full Fine-tuning (Discovery)..."
echo "Output Directory: ${DIR_STEP1}/${GAME_NAME}"
echo "Backbone: ${BACKBONE} | LR: ${LEARNING_RATE}"

python ${TRAIN_SCRIPT} \
    --game_name ${GAME_NAME} \
    --output_dir ${DIR_STEP1} \
    --backbone ${BACKBONE} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --do_preprocess

echo "[Step 2/3] Analyzing Checkpoints & Generating Schedule..."

STEP1_MODEL_PATH="${DIR_STEP1}/${GAME_NAME}"

python ${EXPLORE_SCRIPT} \
    --base_model_dir ${STEP1_MODEL_PATH}

echo "Searching for generated schedule file..."
SCHEDULE_FILE=$(find ${DIR_ANALYSIS} -name "finetuning_schedule.json" | grep "${GAME_NAME}" | grep "metric_l2" | head -n 1)

if [ -z "$SCHEDULE_FILE" ]; then
    echo " Error: Schedule file not found! Analysis might have failed."
    exit 1
else
    echo " Schedule File Found: ${SCHEDULE_FILE}"
fi


echo "[Step 3/3] Starting Optimized Fine-tuning with Schedule..."
echo "Schedule File: ${SCHEDULE_FILE}"
echo "Output Directory: ${DIR_STEP2}/${GAME_NAME}"

python ${TRAIN_SCRIPT} \
    --game_name ${GAME_NAME} \
    --output_dir ${DIR_STEP2} \
    --backbone ${BACKBONE} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --finetune_schedule_file "${SCHEDULE_FILE}" \
    --do_preprocess


echo "========================================================"
echo "ResNet Pipeline Completed Successfully!"
echo "Final Results: ${DIR_STEP2}/${GAME_NAME}"
echo "========================================================"
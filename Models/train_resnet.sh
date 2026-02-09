#!/bin/bash
set -e
# --- 配置区域 (Configuration) ---
GAME_NAME="gallery"
# ResNet 比较轻量，可以尝试大一点的 Batch Size
BATCH_SIZE=4
# ResNet 通常使用比 Transformer 稍大的学习率
LEARNING_RATE=5e-5
NUM_EPOCHS=14
BACKBONE="resnet50"
# 定义输出目录结构 (区分于 ViViT 的结果)
DIR_STEP1="Results/new5/resnet_full"        # Step 1: 全量微调
DIR_ANALYSIS="Results/new5/resnet_exploration" # Step 2: 分析结果
DIR_STEP2="Results/new5/resnet_ours_L2"        # Step 3: 优化微调

# Python 脚本文件名
TRAIN_SCRIPT="defense_resnet.py"
# 分析脚本通常是通用的，只要它能读取 checkpoint 里的 state_dict
EXPLORE_SCRIPT="exploration_resnet_v2.py"

echo "========================================================"
echo "Pipeline Started for Game: ${GAME_NAME} (Backbone: ${BACKBONE})"
echo "========================================================"

# --------------------------------------------------------
# Step 1: Full Fine-tuning (Discovery Run)
# 目的：全量微调，保存每个 Epoch 的 Checkpoints 供分析。
# --------------------------------------------------------
# echo ""
# echo "[Step 1/3] Starting Full Fine-tuning (Discovery)..."
# echo "Output Directory: ${DIR_STEP1}/${GAME_NAME}"
# echo "Backbone: ${BACKBONE} | LR: ${LEARNING_RATE}"

# python ${TRAIN_SCRIPT} \
#     --game_name ${GAME_NAME} \
#     --output_dir ${DIR_STEP1} \
#     --backbone ${BACKBONE} \
#     --num_epochs ${NUM_EPOCHS} \
#     --batch_size ${BATCH_SIZE} \
#     --learning_rate ${LEARNING_RATE} \
#     --do_preprocess \

echo "[Step 2/3] Analyzing Checkpoints & Generating Schedule..."
# 模型的输入路径 (Step 1 的输出)
STEP1_MODEL_PATH="${DIR_STEP1}/${GAME_NAME}"
# 注意：对于 ResNet，exploration.py 的 model_ckpt 参数可能只是个占位符，
# 或者是用来标记日志的，具体取决于 exploration.py 的实现。
# 这里我们传入 resnet50 以防万一。
python ${EXPLORE_SCRIPT} \
    --base_model_dir ${STEP1_MODEL_PATH} \
    --output_base_dir ${DIR_ANALYSIS}

# --- 自动定位生成的 Schedule 文件 ---
echo "Searching for generated schedule file..."
# 查找包含 metric_avg_l2 的策略文件
SCHEDULE_FILE=$(find ${DIR_ANALYSIS} -name "finetuning_schedule.json" | grep "${GAME_NAME}" | grep "metric_l2" | head -n 1)

if [ -z "$SCHEDULE_FILE" ]; then
    echo "❌ Error: Schedule file not found! Analysis might have failed."
    exit 1
else
    echo "✅ Schedule File Found: ${SCHEDULE_FILE}"
fi

# --------------------------------------------------------
# Step 3: Optimized Fine-tuning (Application Run)
# 目的：使用生成的 Schedule 文件重新训练。
# --------------------------------------------------------

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
set -e

GAME_NAME="solid"
NUM_EPOCHS=14
BATCH_SIZE=4
LEARNING_RATE=1e-3
LORA_R=64
LORA_A=128

echo "========================================================"
echo "Pipeline Started for Game: ${GAME_NAME}"
echo "LORA_R: ${LORA_R}"
echo "LORA_A: ${LORA_A}"
echo "========================================================"

python vivit_lora.py \
    --game_name "${GAME_NAME}" \
    --output_dir Results/new5/lora_${LORA_R}_${LEARNING_RATE} \
    --num_epochs 14 \
    --learning_rate ${LEARNING_RATE} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_A} \
    --do_preprocess
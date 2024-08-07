#!/bin/bash
uname -a
date
export CUDA_LAUNCH_BLOCKING=1

DATA_PATH='/home/tsl/data/HumanParsing/mada-data/single_human_data'
# 打印DATA_PATH数据路径
echo "data path: ${DATA_PATH}"

GPU_NUMS='0'
NUM_CLASSES=20
EVAL_EPOCHS=5
BATCH_SIZE=8
MODEL_RESTORE='./log/checkpoint.pth.tar'
LOG_DIR='./logs/lip_solider_swin_base'
WARMUP_EPOCHS=30
EPOCHS=150
SCHP_START=120
INPUT_SIZE='512,512'
ARCH='swin_base'
IMAGENET_PRETRAIN='./pretrained/solider_swin_base.pth'
LEARNING_RATE=7e-3
WEIGHT_DECAY=0
OPTIMIZER='sgd'
LR_DIVIDER=500
CYCLELR_DIVIDER=2


# 获取new_checkpoint.txt中的内容并拼接
NEW_CHECKPOINT_FILE="${LOG_AND_SAVE_PATH}/new_checkpoint.txt"
if [ -f "${NEW_CHECKPOINT_FILE}" ]; then
    NEW_CHECKPOINT=$(cat ${NEW_CHECKPOINT_FILE})
    RESTORE_FROM="${LOG_AND_SAVE_PATH}/${NEW_CHECKPOINT}"
else
    RESTORE_FROM="${LOG_AND_SAVE_PATH}/checkpoint.pth.tar"
fi

echo "restore from: ${RESTORE_FROM}"

# 设置最大内存使用量为 50GB
# ulimit -v 52428800

# 限制程序可见GPU为GPU_NUMS中的值，并增加提示输出
export CUDA_VISIBLE_DEVICES=${GPU_NUMS}
echo "Using GPU: ${GPU_NUMS}"

python train_local.py \
    --data-dir ${DATA_PATH} \
    --num-classes ${NUM_CLASSES} \
    --input-size ${INPUT_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --gpu ${GPU_NUMS} \
    --epochs ${EPOCHS} \
    --eval-epochs ${EVAL_EPOCHS} \
    --model-restore ${RESTORE_FROM} \
    --arch ${ARCH} \
    --imagenet-pretrain ${IMAGENET_PRETRAIN} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --optimizer ${OPTIMIZER} \
    --syncbn \
    --lr_divider ${LR_DIVIDER} \
    --cyclelr-divider ${CYCLELR_DIVIDER} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --schp-start ${SCHP_START} \
    --log-dir ${LOG_DIR} \
    --model-restore ${MODEL_RESTORE}
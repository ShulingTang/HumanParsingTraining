#!/bin/bash

# 获取脚本开始时间
start_time=$(date +%s)

# 定义路径log
log_path="./logs/_solider_swin_base"
output_file="${log_path}/new_checkpoint.txt"
# 最终检查点文件名
checkpoint_end="${log_path}/checkpoint_20.pth.tar"

# 训练脚本文件名
train_bash_file="train.sh"

# 检查train.sh是否存在且可执行
if [ ! -f "$train_bash_file" ]; then
    echo "脚本文件 $train_bash_file 不存在！"
    exit 1
fi

if [ ! -x "$train_bash_file" ]; then
    echo "脚本文件 $train_bash_file 不可执行！"
    exit 1
fi

# 监控并运行train.sh脚本
while true; do

    # 获取number数值最大的文件名
    max_file=$(ls "$log_path"/checkpoint_*.pth.tar 2>/dev/null | awk -F'[_ .]' '{print $(NF-2)}' | sort -n | tail -1)
    if [ -z "$max_file" ]; then
        max_filename="checkpoin_null.pth.tar"
    else
        max_filename="checkpoint_${max_file}.pth.tar"
    fi

    if [ "$max_filename" == "$checkpoint_end" ]; then
        # 计算脚本运行时间
        end_time=$(date +%s)
        runtime=$((end_time - start_time))

        # 转换运行时间为小时、分钟和秒
        hours=$((runtime / 3600))
        minutes=$(((runtime % 3600) / 60))
        seconds=$((runtime % 60))

        # 输出运行时间和“训练完成”
        echo "脚本运行时间：${hours}小时${minutes}分钟${seconds}秒"
        echo "训练完成"
        exit 0
    else
        # 写入最大文件名到new_checkpoint.txt
        if [ -f "$output_file" ]; then
            rm "$output_file"
        fi
        echo "$max_filename" > "$output_file"
    fi
    # 再次运行train.sh脚本
    ./$train_bash_file

done

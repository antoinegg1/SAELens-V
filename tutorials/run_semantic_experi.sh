#!/usr/bin/env bash

# ============ 配置部分 ============
# 数据集所在目录，里面包含若干 imagenet10_*_dataset 这样的子目录
DATASETS_DIR="/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data"

# 用于存放推理结果的目录
OUTPUT_DIR="/aifs4su/yaodong/changye/data/semantic_result"

# 你的推理脚本，即前面写好的基于 llava 的 Python 脚本
INFERENCE_SCRIPT="/aifs4su/yaodong/changye/SAELens-V/tutorials/semantic_experi.py"

# 创建输出目录（若不存在）
mkdir -p "$OUTPUT_DIR"

# ============ 批量推理逻辑 ============
# 遍历 datasets_dir 下以 imagenet10_ 开头的文件夹
for dataset_path in "$DATASETS_DIR"/imagenet1k_*; do
    # 判断确实是文件夹才进行处理
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        save_path="$OUTPUT_DIR/${dataset_name}_result"

        echo "Processing dataset: $dataset_name"
        echo "  Input:  $dataset_path"
        echo "  Output: $save_path"

        # 调用推理脚本
        python "$INFERENCE_SCRIPT" \
            --data_set "$dataset_path" \
            --save_path "$save_path"

        echo "---------------------------------------"
    fi
done

echo "All datasets processed successfully."

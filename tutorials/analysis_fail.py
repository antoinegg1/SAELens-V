#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from datasets import load_from_disk
from collections import Counter

def analyze_failures_and_count(results_dir):
    """
    1. 在 results_dir 下寻找所有 *_result 目录，并加载里面的推理数据集；
    2. 对每个数据集找到失败案例的 index，写到 fail_cases.txt 中；
    3. 汇总所有数据集的失败次数到 global_fail_counts.txt。
    """

    # 1) 找到所有 *_result 目录
    result_folders = sorted(glob.glob(os.path.join(results_dir, "*_result")))

    if not result_folders:
        print(f"在 {results_dir} 下没有找到 *_result 结尾的文件夹，请检查路径或命名。")
        return

    # 用于全局统计：哪一个 index 在多少个数据集出现 fail（或出现多少次 fail）
    # 这里计数器可按【index -> fail 次数】累计
    fail_counter_global = Counter()

    print(f"发现 {len(result_folders)} 个结果文件夹，开始分析...\n")

    for folder in result_folders:
        print(f"===== Analyzing: {folder} =====")

        # 加载推理后数据集
        ds = load_from_disk(folder)

        # 检查列是否存在
        if "label" not in ds.column_names or "prediction" not in ds.column_names:
            print(f"  [警告] {folder} 中不包含 label 或 prediction 列，跳过。")
            continue

        # 找到 fail 的样本编号
        fails = []
        for idx in range(len(ds)):
            sample = ds[idx]
            true_label = str(sample["label"])
            pred_label = str(sample["prediction"])
            if true_label != pred_label:
                fails.append(idx)  # 记录失败的 index

        fail_count = len(fails)
        total_count = len(ds)
        accuracy = 1 - fail_count / total_count if total_count else 0.0

        print(f"  Total samples: {total_count}")
        print(f"  Fail count:    {fail_count}")
        print(f"  Accuracy:      {accuracy:.4f}")

        # 将失败编号写入该数据集的 fail_cases.txt
        fail_txt_path = os.path.join(folder, "fail_cases.txt")
        with open(fail_txt_path, "w", encoding="utf-8") as f:
            for idx in fails:
                f.write(str(idx) + "\n")  # 每行一个编号
        print(f"  -> 失败案例编号已写入: {fail_txt_path}")

        # 更新全局 fail 计数器
        # 逻辑：对于同一个 index，失败 1 次就 +1
        for idx in fails:
            fail_counter_global[idx] += 1

        print()

    # ============ 统计全局 fail 次数并输出 ============
    # 如果所有数据集都应该有相同数量的样本，那么每个 index 都在 0 ~ (N-1) 范围内
    # 这里把 fail_counter_global 里的 index 按次数从高到低排序
    sorted_fail_counts = sorted(fail_counter_global.items(), key=lambda x: x[1], reverse=True)

    # 写一个 global_fail_counts.txt 放在 results_dir 下
    global_fail_path = os.path.join(results_dir, "global_fail_counts.txt")
    with open(global_fail_path, "w", encoding="utf-8") as f:
        f.write("index,fail_count\n")
        for idx, count in sorted_fail_counts:
            f.write(f"{idx},{count}\n")

    print(f"所有数据集分析完成，总的 fail 计数写入: {global_fail_path}")

def main():
    # 你可以在这里写死路径，或用 argparse 做命令行传参
    # 例如:
    results_dir = "/aifs4su/yaodong/changye/data/semantic_result"
    analyze_failures_and_count(results_dir)

if __name__ == "__main__":
    main()

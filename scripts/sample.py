import argparse
import json
import random

from datasets import load_dataset, Dataset
import random
import os
def sample_from_json(json_path, sample_size,output_path):
    # 加载 JSON 文件
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # 检查样本数量并抽样
    if sample_size > len(data):
        print(f"样本数量大于数据总量，将返回全部数据。")
        sample_size = len(data)
    
    sampled_data = random.sample(data, sample_size)
    
    # 输出采样结果
    print(f"从 JSON 文件中抽取了 {sample_size} 条样本数据。")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(sampled_data, file, ensure_ascii=False, indent=4)
    return sampled_data


def sample_and_save_dataset_as_hf(
    dataset_name: str,
    split: str = "train",
    sample_size: int = 1000,
    seed: int = 42,
    output_dir: str = "sampled_dataset"
):
    """
    从 Hugging Face datasets 中加载数据，抽样 sample_size 条样本并保存为 HF 的 Dataset 格式。
    
    Args:
        dataset_name (str): 数据集名称，例如 "imdb" 或 "your_dataset/script_path.py"
        split (str): 要加载的划分，例如 "train"、"test" 或 "train[:10%]"
        sample_size (int): 抽样数量
        seed (int): 随机种子
        output_dir (str): 输出目录，用于保存 Dataset
    """
    # 加载数据集
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 抽样
    if sample_size > len(dataset):
        raise ValueError(f"样本数超出数据集大小（{len(dataset)}）")

    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    # 保存为 HF Dataset 格式
    sampled_dataset.save_to_disk(output_dir)
    print(f"Sampled {sample_size} samples and saved to {output_dir} as Hugging Face Dataset format.")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Sample data from a JSON file.")
    # parser.add_argument("--json_path", type=str, help="JSON 文件路径")
    # parser.add_argument("--sample_size", type=int, help="抽样数量")
    # parser.add_argument("--output_path", type=str, help="输出文件路径")
    
    # args = parser.parse_args()
    
    # # 调用抽样函数
    # sampled_data = sample_from_json(args.json_path, args.sample_size,args.output_path)
    # print(sampled_data)
    sample_and_save_dataset_as_hf(
    dataset_name="/aifs4su/yaodong/changye/data/AA_preference",
    split="train",
    sample_size=1000,
    output_dir="AA_preference1k"
)


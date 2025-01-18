import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset,load_from_disk
import hashlib
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from collections import Counter
import argparse
import os

def regularize_Compcap(data_path):
    # 读取数据集
    dataset = load_dataset(data_path)
    # 选择需要保留的列
    columns_to_keep = ["prompt", "Image", "conversation"]
    # 使用 remove_columns 移除不需要的列
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])

    return dataset
    

def generate_text_hash(text: str) -> str:
    """
    Generate a unique identifier for the given text using SHA-256.

    Args:
        text (str): Input text.

    Returns:
        str: Unique hash for the text.
    """
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()
# 示例文件路径
file_path = "/mnt/file2/changye/dataset/Align-Anything-preference_interp/Align-Anything-preference_coocur.pt"

os.environ['HF_DATASETS_CACHE']='/mnt/file2/changye/tmp'
# 读取 .pt 文件
data = torch.load(file_path)

dataset_path="/mnt/file2/changye/dataset/Align-Anything_preference"
train_dataset = load_dataset(
            dataset_path,
            split="train",
            trust_remote_code=True,
        )
# 定义需要保留的列
columns_to_keep = ["prompt", "Image", "conversation"]

# 使用 remove_columns 移除不需要的列
train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_keep])

data_list=[]
for data_l in data:
    keys=list(data_l.keys())
    for key in keys:
        data_list.append((key,data_l[key]))

dataset_dict = {}
for j, item in tqdm.tqdm(enumerate(train_dataset), desc="Processing data_list", unit="item"):
    key = generate_text_hash(item['prompt']+item['response_1']+item['response_2'])
    if key not in dataset_dict:
        dataset_dict[key] = []
    dataset_dict[key].append(j)  # 记录每个idx对应的所有匹配位置

# 2. 生成 formatted_dataset，确保按照顺序匹配
formatted_dataset = []
index_set = set()  # 用来记录已经匹配过的 train_dataset 索引

# 使用 tqdm 显示进度条
for i in tqdm.tqdm(range(len(data_list)), desc="Processing data_list", unit="item"):
    key_datalist, value_datalist = data_list[i]
    
    # 查找是否存在匹配的key
    if key_datalist in dataset_dict:
        # 遍历匹配的索引
        for j in dataset_dict[key_datalist]:
            if j not in index_set:  # 如果该项未匹配过
                # 如果匹配，则将 value 从 data_list 加到 train_dataset 项中
                new_item = train_dataset[j].copy()  # 拷贝 train_dataset 项
                new_item["coocur"] = value_datalist
                formatted_dataset.append(new_item)
                
                # 将已匹配的索引添加到 index_set
                index_set.add(j)
                break  # 找到一个匹配就跳出内层循环，继续下一个 data_list 的元素
# 定义文件路径
cosi_file_path = "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_cosi_weight/cosi_feature_list.txt"  # 将此替换为你的文件路径

# 读取文件并转换为字典
osi_dict = {}
with open(cosi_file_path, "r") as file:
    for line in file:
        key, value = line.strip().split(",")  # 按逗号分割每行
        osi_dict[int(key)] = float(value)    # 将 key 转为 int, value 转为 float

# 输出结果
print(osi_dict)
set_cosi_key=set(osi_dict.keys())
cosi_coocur_data={}
for f_data in tqdm.tqdm(formatted_dataset):
    score=0
    for value in f_data['coocur']:
        if value in set_cosi_key:
            score+=osi_dict[value]
    f_data["Cooccur_score"]=score
dataset = Dataset.from_list(formatted_dataset)
dataset.save_to_disk("/data/changye/data/Align-Anything-cosi-full")
for item in formatted_dataset:
    item["l0"]=float(item["l0"])
formatted_dataset_sorted = sorted(formatted_dataset, key=lambda x: x["Cooccur_score"], reverse=True)


# 2. 将数据集分成四个分位
num_samples = len(formatted_dataset_sorted)
q1 = int(num_samples * 0.25)
q2 = int(num_samples * 0.5)
q3 = int(num_samples * 0.75)

# 将数据集切分为四个分位
split_datasets = {
    "q0_25": formatted_dataset_sorted[:q1],
    "q25_50": formatted_dataset_sorted[q1:q2],
    "q50_75": formatted_dataset_sorted[q2:q3],
    "q75_100": formatted_dataset_sorted[q3:]
}

# 3. 转换为 HuggingFace Dataset 格式并保存，显示进度条
hf_datasets = {}
for split_name, split_data in tqdm.tqdm(split_datasets.items(), desc="Processing splits", unit="split"):
     # 初始化一个空字典用于存储格式化数据
    formatted_data = {}
    
    # 添加进度条，逐列转换数据
    for key in tqdm.tqdm(split_data[0].keys(), desc=f"Formatting {split_name}", unit="column",position=1):
        formatted_data[key] = [d[key] for d in split_data]
    
    # 将格式化的数据创建为 HuggingFace 数据集
    hf_datasets[split_name] = Dataset.from_dict(formatted_data)

    # 保存每个分位数据集到磁盘
    hf_datasets[split_name].save_to_disk(f"./{split_name}_dataset")

# 输出一下分割的数据集
for split_name, dataset in hf_datasets.items():
    print(f"{split_name} dataset:")
    print(dataset)


# 假设 data_list 是一个包含数值的列表
# 例如：
# data_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# values_list = [tensor['Cooccur'].cpu().item() if tensor['Cooccur'].is_cuda else tensor['Cooccur'].item() for tensor in formatted_dataset_sorted]
values_list=[float(tensor["Cooccur_score"]) for tensor in formatted_dataset_sorted]
# 统计频率
num_bins = 10

# 生成直方图
frequencies, bin_edges = np.histogram(values_list, bins=num_bins)

# 计算每个区间的中心点（用于插值）
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 使用样条插值将直方图转换为平滑曲线
bin_centers_smooth = np.linspace(bin_centers[0], bin_centers[-1], 300)  # 插值点
frequencies_smooth = make_interp_spline(bin_centers, frequencies)(bin_centers_smooth)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(bin_centers_smooth, frequencies_smooth, color='orange', lw=2)


# 样式设置
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel('Cosimilarity score of data ', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('The distribution of Align-Anything preference data based on Cosimilarity score', fontsize=16)
plt.legend(fontsize=12)
plt.show()

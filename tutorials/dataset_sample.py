from datasets import load_dataset, DatasetDict

# 加载数据集
dataset = load_dataset("/aifs4su/yaodong/changye/data/flickr30k")

# 随机采样
sample_size = 3000
sampled_dataset = dataset['test'].shuffle(seed=42).select(range(sample_size))

# 重新保存采样后的数据集
sampled_dataset.save_to_disk("/aifs4su/yaodong/changye/data/flickr3k")


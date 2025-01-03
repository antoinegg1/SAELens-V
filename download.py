from datasets import load_dataset

# 设置缓存目录
cache_dir = "/mnt/file2/changye/dataset/Align-Anything_preference"

# 加载指定的子数据集
train_dataset = load_dataset(
    'PKU-Alignment/Align-Anything',
    name='text-image-to-text',  # 子数据集的名称
    cache_dir=cache_dir         # 本地缓存目录
)['train']

# 查看数据集信息
print(train_dataset)


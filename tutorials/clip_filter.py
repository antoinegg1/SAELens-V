import argparse
import os
import torch
import clip
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from hashlib import sha256
from torch.utils.data import Dataset
from tqdm import tqdm

# 生成哈希（文本内容指纹）
def generate_text_hash(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()

# 格式化样本，提取 prompt、图像、hash
def format_Anything_sample(raw_sample: dict):
    system_prompt = ""
    user_prompt = 'USER: \n<image> {input}'
    assistant_prompt = '\nASSISTANT: {output}'

    prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
    image = Image.open(BytesIO(raw_sample['image'])).resize((336, 336)).convert('RGB')
    text_hash = generate_text_hash(raw_sample['question'] + raw_sample['response_1'] + raw_sample['response_2'])

    formatted_prompt = f'{system_prompt}{user_prompt.format(input=prompt)}{assistant_prompt.format(output="")}'
    return {'prompt': formatted_prompt, 'image': image, 'text_hash': text_hash}

# 包装成 PyTorch Dataset，用于 CLIP 特征提取（顺序与原数据集保持一致）
class FilterableCLIPDataset(Dataset):
    def __init__(self, hf_dataset, preprocess):
        self.dataset = hf_dataset
        self.preprocess = preprocess
        print("格式化样本中...")
        self.formatted_data = [format_Anything_sample(sample) for sample in tqdm(self.dataset, desc="格式化数据")]
        print(f"样本格式化完成，共 {len(self.formatted_data)} 条样本。")

    def __len__(self):
        return len(self.formatted_data)

    def __getitem__(self, idx):
        item = self.formatted_data[idx]
        image_tensor = self.preprocess(item['image'])
        prompt = item['prompt']
        return image_tensor, prompt, item['text_hash']

def main():
    parser = argparse.ArgumentParser(description="CLIP分数10分位数据切割并保存")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="CLIP模型名称或路径")
    parser.add_argument("--dataset_name", type=str, default="/aifs4su/yaodong/changye/data/AA_preference", help="Huggingface数据集名称")
    parser.add_argument("--split", type=str, default="train", help="数据集切分名称，例如 train/validation")
    parser.add_argument("--sample_percentage", type=float, default=1.0, help="使用数据集的百分比 (例如 0.05 表示 5%)")
    parser.add_argument("--save_dir", type=str, default="/aifs4su/yaodong/changye/data/AA_preference_clip", help="输出保存路径")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("加载 CLIP 模型中...")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    print(f"模型加载完成，使用设备: {device}")

    # 加载 Huggingface 数据集
    split_str = args.split if args.sample_percentage >= 1.0 else f"{args.split}[:{int(args.sample_percentage * 100)}%]"
    print(f"加载 Huggingface 数据集：{args.dataset_name} [{split_str}]")
    hf_dataset = load_dataset(args.dataset_name, split=split_str)
    print(f"加载完成，样本数：{len(hf_dataset)}")

    # 构造用于特征提取的 PyTorch Dataset
    clip_dataset = FilterableCLIPDataset(hf_dataset, preprocess)

    # 提取图文特征
    print("提取图文特征中...")
    image_features, text_features = [], []
    with torch.no_grad():
        for idx in tqdm(range(len(clip_dataset)), desc="提取特征"):
            image_tensor, prompt, _ = clip_dataset[idx]
            image_input = image_tensor.unsqueeze(0).to(device)
            image_feature = model.encode_image(image_input).cpu()
            image_features.append(image_feature)

            tokenized = clip.tokenize(prompt).to(device)
            text_feature = model.encode_text(tokenized).cpu()
            text_features.append(text_feature)

    print("特征提取完成，计算相似度中...")
    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features, dim=1)
    logit_scale = model.logit_scale.exp().item()
    clip_scores = cos_sim * logit_scale

    # 切分并保存数据
    print("根据 CLIP 相似度进行10分位切割并保存...")
    for i in tqdm(range(10), desc="切割与保存"):
        lower_quantile = i / 10
        upper_quantile = (i + 1) / 10
        lower_bound = torch.quantile(cos_sim.float(), lower_quantile)
        upper_bound = torch.quantile(cos_sim.float(), upper_quantile)

        if i < 9:
            indices = torch.where((cos_sim >= lower_bound) & (cos_sim < upper_bound))[0]
        else:
            indices = torch.where((cos_sim >= lower_bound) & (cos_sim <= upper_bound))[0]

        group_name = f"top_{int(lower_quantile*100)}_{int(upper_quantile*100)}"
        indices = indices.tolist()
        print(f" - {group_name}: {len(indices)} samples")

        subset = hf_dataset.select(indices)
        save_path = os.path.join(args.save_dir, group_name)
        subset.save_to_disk(save_path)
        print(f" ✔ 子数据集 {group_name} 保存至: {save_path}")

    print("✅ 全部分位数据保存完成！")

if __name__ == "__main__":
    main()

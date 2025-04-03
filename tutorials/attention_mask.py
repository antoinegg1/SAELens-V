import torch
import numpy as np
from PIL import Image
from datasets import load_from_disk

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# 0. 加载数据集（假设是HF datasets格式）
dataset_path = "/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data/imagenet1k_10"
dataset = load_from_disk(dataset_path)  # 如果有多个split，如 train/test，可分别处理

# 1. 加载模型和处理器
model_path = "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # 可改成 torch.float16 若你的GPU支持
    low_cpu_mem_usage=True
)
model.to("cuda:0")
model.eval()

# 2. 定义新的 prompt
example_prompt = """Analyze the given image and classify it into one of the labels below.

Labels:
0: bonnet, poke bonnet
1: green mamba
2: langur
3: Doberman, Doberman pinscher
4: gyromitra
5: Saluki, gazelle hound
6: vacuum, vacuum cleaner
7: window screen
8: cocktail shaker
9: garden spider, Aranea diademata

Your response must contain only the corresponding label number. No explanations, no extra text."""

# 3. 函数：将指定的 patch 直接涂黑
def mask_image_patches(
    img: Image.Image,
    patch_indices: np.ndarray,
    patch_size: int = 14,  # 如果模型用336x336 + 14x14 patch
    grid_size: int = 24
) -> Image.Image:
    """
    将图像 img(336×336) 拆分为 grid_size×grid_size=24×24=576 个 14×14 大小的 patch，
    对于 patch_indices 中的 patch，直接在图像中涂黑。
    """
    img = img.convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    H, W, C = img_np.shape

    # 检查图像大小是否匹配
    assert H == 336 and W == 336, f"图像尺寸=({H},{W})，需与(336,336)匹配"
    assert H == patch_size * grid_size, "patch_size×grid_size != 图像大小"

    for patch_idx in patch_indices:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        r_start = row * patch_size
        c_start = col * patch_size
        img_np[r_start:r_start+patch_size, c_start:c_start+patch_size, :] = 0

    return Image.fromarray(img_np)

# 4. 处理每个样本的函数：计算后25%注意力的patch，生成mask后图像，替换 image 列
def process_example(example):
    # example["image"] 应该是一个 PIL.Image
    pil_img = example["image"]

    # 如果模型是 336×336 输入，但当前图片大小不是 336×336，需要 resize
    img_resized = pil_img.resize((336, 336))

    # 构造对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": example_prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # 做一次前向，获取注意力
    inputs = processor(
        images=img_resized,
        text=prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.vision_tower(
            pixel_values=inputs["pixel_values"][:, 0, :, :, :],
            output_attentions=True
        )

    # 拿倒数第2层注意力
    hook_layer_idx = -2
    attn = outputs.attentions[hook_layer_idx]  # shape (1, heads, seq_len, seq_len)

    # 对多头做平均，取 batch=0
    attn_avg = attn.mean(dim=1)[0]  # (seq_len, seq_len)
    cls_attn = attn_avg[0, 1:].cpu().numpy()  # 去掉 CLS token (索引0)

    # 找后25% patch
    threshold = np.percentile(cls_attn, 75)
    low_attn_indices = np.where(cls_attn <= threshold)[0]

    # 将相应的 patch 在图像中涂黑
    masked_img = mask_image_patches(
        img_resized,
        low_attn_indices,
        patch_size=14,
        grid_size=24
    )

    # 用被mask后的图像替换原来的 image
    example["image"] = masked_img
    return example

# 5. 对整个数据集映射，获得新的数据集
#    如果你的数据集很大，启用 batch 或多进程，需要额外设置。这里做单进程示例:
processed_dataset = dataset.map(process_example, num_proc=1)

# 6. 保存处理完的数据集
save_path = "/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data/imagenet10_attn_25"
processed_dataset.save_to_disk(save_path)

print("处理完成，新的数据集已保存至:", save_path)

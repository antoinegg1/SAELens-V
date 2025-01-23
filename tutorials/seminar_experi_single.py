import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sae_lens.activation_visualization import load_llava_model,load_sae,generate_with_saev
# 环境配置
os.environ['HF_HOME'] = "/mnt/file2/changye/tmp/"
os.environ['TMPDIR'] = "/mnt/file2/changye/tmp/"
os.environ['HF_DATASETS_CACHE'] = "/mnt/file2/changye/tmp/"
os.environ['https_proxy'] = "127.0.0.1:7895"

# 请替换为自己的路径
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = "/mnt/file2/changye/model/llava"
device = "cuda:0"
sae_device = "cuda:7"
sae_path = "/mnt/file2/changye/model/SAE/llavasae_obliec100k_SAEV"

save_path = "../activation_visualization"

# 加载模型
processor, hook_language_model = load_llava_model(MODEL_NAME, model_path, device, n_devices=8)
sae = load_sae(sae_path, sae_device)

# 输入的 prompt
example_prompt = """You are provided with an image and a list of 10 possible labels. Your task is to classify the image by selecting the most appropriate label from the list below:

Labels:
0: "bonnet, poke bonnet"
1: "green mamba"
2: "langur"
3: "Doberman, Doberman pinscher"
4: "gyromitra"
5: "Saluki, gazelle hound"
6: "vacuum, vacuum cleaner"
7: "window screen"
8: "cocktail shaker"
9: "garden spider, Aranea diademata"

Carefully analyze the content of the image and identify which label best describes it. Then, output only the **corresponding number** from the list without any additional text or explanation.
"""

# 图像处理函数：根据激活值裁剪图像
def crop_image_by_activation(image, activation, keep_ratio):
    """根据激活值裁剪图像，保留指定比例的激活区域，剩余区域填充为黑色"""
    image = image.resize((336, 336)).convert('RGB')  # 确保为RGB三通道
    image_array = np.array(image)  # (H, W, C)
    # breakpoint()
    # 获取图像的高度和宽度
    h, w, c = image_array.shape
    activation = activation.reshape((24, 24))
    activation_size = activation.shape  # 激活图的尺寸 (24, 24)
    
    # 计算每个activation的区域对应图像的比例
    patch_size_h = h // activation_size[0]
    patch_size_w = w // activation_size[1]
    
    # 计算保留的激活区域数量
    keep_count = int(activation_size[0] * activation_size[1] * keep_ratio)
    
    # 根据激活值选择保留区域
    flattened_activation = activation.flatten()
    threshold = np.partition(flattened_activation, -keep_count)[-keep_count]  # 找出需要保留的激活值的阈值
    
    mask = activation >= threshold  # 创建一个掩码，保留大于等于阈值的区域
    mask = mask.reshape(activation_size)  # 将掩码恢复为原来的激活图尺寸

    # 根据掩码保留对应的图像区域，其他区域填充为黑色
    cropped_image = np.zeros_like(image_array)
    for i in range(activation_size[0]):
        for j in range(activation_size[1]):
            if mask[i, j]:
                cropped_image[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w, :] = image_array[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w, :]
    
    return Image.fromarray(cropped_image.astype(np.uint8))

# 准备输入，结合prompt和图片
def prepare_inputs(prompt, image, processor):
    image = image.resize((336, 336)).convert('RGBA')
    formatted_prompt = f"{prompt}<image>"  # 只使用prompt
    text_input = processor.tokenizer(formatted_prompt, return_tensors="pt")
    image_input = processor.image_processor(images=image, return_tensors="pt")
    return {
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "pixel_values": image_input["pixel_values"],
        "image_sizes": image_input["image_sizes"],
    }

# 生成图片和激活值
def generate_images(image, inputs):
    """生成原图和三个不同保留比例的图像"""
    original_image = image.resize((336, 336))  # 原图

    data = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
        "image_sizes": inputs["image_sizes"].to(device),
    }

    # 生成输出
    total_activation_l0_norms_list, patch_features_list, feature_act_list, image_indice, output = generate_with_saev(
        data, hook_language_model, processor, save_path, image, sae, sae_device, max_new_tokens=1, selected_feature_indices=None,
    )

    activation_l0 = total_activation_l0_norms_list[0]  # 获取激活层的数值

    # 为不同的激活比例生成图像
    image_25 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.25)
    image_50 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.50)
    image_75 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.75)

    image_list = [
        {"image": original_image, "name": "original"},
        {"image": image_25, "name": "25_percent"},
        {"image": image_50, "name": "50_percent"},
        {"image": image_75, "name": "75_percent"},
    ]

    return image_list

# 读取输入图片
image_path = "/mnt/file2/changye/SAELens-V-result/result/semantic_image/mustimi.png"
image = Image.open(image_path)

# 准备输入
inputs = prepare_inputs(example_prompt, image, processor)

# 生成四张图像：原图和不同激活值保留比例的图像
image_list = generate_images(image, inputs)

# 创建保存文件夹
save_dir = "/mnt/file2/changye/processed_images"
os.makedirs(save_dir, exist_ok=True)

# 保存四张图像

for item in image_list:
    item["image"].save(os.path.join(save_dir, f"{item['name']}.png"))

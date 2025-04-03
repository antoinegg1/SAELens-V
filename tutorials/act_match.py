from datasets import load_from_disk
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sae_lens.activation_visualization import load_llava_model, load_sae, generate_with_saev

# ========== 配置路径 ==========
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf"
device = "cuda:5"
sae_device = "cuda:7"
sae_path = "/aifs4su/yaodong/changye/model/Antoinegg1/llavasae_obliec100k_SAEV"
dataset_path = "/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data/imagenet1k_10"
save_path = "/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data/imagenet1k_10_test"
cosi_file_path = "/aifs4su/yaodong/changye/data/AA_preference_mistral-7b_interp/AA_preference_cosi_weight/cosi_feature_list.txt"

# ========== Prompt ==========
example_prompt = """You are provided with an image and a list of 10 possible labels..."""  # 可替换为完整文本
system_prompt = " "
user_prompt = 'USER: \n<image> {input}'
assistant_prompt = '\nASSISTANT: {output}'

# ========== 函数定义 ==========

def load_model_and_sae():
    processor, hook_lm = load_llava_model(MODEL_NAME, model_path, device, n_devices=2)
    sae = load_sae(sae_path, sae_device)
    return processor, hook_lm, sae

def prepare_inputs(prompt, image, processor):
    """对图像和文本进行预处理，生成模型输入"""
    image = image.resize((336, 336)).convert('RGBA')
    formatted_prompt = f'{system_prompt}{user_prompt.format(input=prompt)}{assistant_prompt.format(output="")}'
    text_input = processor.tokenizer(formatted_prompt, return_tensors="pt")
    image_input = processor.image_processor(images=image, return_tensors="pt")
    return {
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "pixel_values": image_input["pixel_values"],
        "image_sizes": image_input["image_sizes"],
    }

def load_feature_indices(filepath):
    """加载 COSI 特征索引列表"""
    osi_list = []
    with open(filepath, "r") as file:
        for line in file:
            key, value = line.strip().split(",")
            osi_list.append((int(key), float(value)))
    return osi_list

def extract_active_patches(patch_indices, feature_acts, target_feature):
    """提取激活了特定特征的 image patch 索引"""
    act_set = set()
    for idx in tqdm.tqdm(patch_indices):
        if feature_acts[0][0][idx][target_feature] != 0:
            if idx > 582:
                act_set.add(idx - 583)
            else:
                act_set.add(idx)
    return act_set

def visualize_patch_activation(image, activated_patches, feature_acts, target_feature):
    """可视化图像上某个特征在不同 patch 上的激活区域"""
    H, W, patch_count = 336, 336, 24
    patch_size = H // patch_count
    img = np.array(image.resize((H, W)))

    if img.ndim == 2:  # 灰度图转 RGB
        img = np.stack([img] * 3, axis=-1)

    highlight = np.zeros_like(img, dtype=np.float32)
    highlight[..., :2] = 255  # yellow
    alpha = 0.3

    for i in range(patch_count):
        for j in range(patch_count):
            patch_idx = i * patch_count + j
            if patch_idx in activated_patches:
                if feature_acts[0][0][patch_idx][target_feature] > 5 or feature_acts[0][0][patch_idx + 583][target_feature] > 5:
                    patch_area = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = (
                        patch_area * (1 - alpha) + highlight[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] * alpha
                    ).astype(np.uint8)

    display(Image.fromarray(img.astype(np.uint8)))  # Jupyter 环境下显示

# ========== 主流程 ==========

def main(index=370, target_feature=44031):
    print(">> 加载模型和SAE...")
    processor, hook_lm, sae = load_model_and_sae()

    print(">> 加载数据集和特征...")
    dataset = load_from_disk(dataset_path)
    cosi_features = load_feature_indices(cosi_file_path)

    print(">> 准备输入数据...")
    inputs = prepare_inputs(example_prompt, dataset[index]['image'], processor)

    model_input = {
        k: v.to(device) for k, v in inputs.items()
    }

    print(">> 运行 generate_with_saev 获取激活...")
    total_activation_l0_norms_list, patch_indices_list, feature_acts_list = generate_with_saev(
        model_input, hook_lm, processor, save_path, dataset["image"][index],
        sae, sae_device, max_new_tokens=1, selected_feature_indices=cosi_features, strategy="all"
    )

    print(">> 提取激活的 patch 索引...")
    active_patches = extract_active_patches(patch_indices_list[0], feature_acts_list, target_feature)

    print(">> 可视化特征激活区域...")
    visualize_patch_activation(dataset[index]['image'], active_patches, feature_acts_list, target_feature)

if __name__ == "__main__":
    main()

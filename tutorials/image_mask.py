from datasets import load_from_disk
from PIL import Image
import numpy as np
from tqdm import tqdm

# 加载数据集
data = load_from_disk("/aifs4su/yaodong/changye/data/OKVQA_cosi")

def process_image_mask(sample):
    # === 获取图像和激活 ===
    image = sample["image"]
    patch_induce_list = sample["patch_induce_list"][0]
    activate_list = sample["activate_list"]
    
    selected_acts = [activate_list[i] for i in patch_induce_list]
    selected_acts = np.array(selected_acts)
    
    if len(selected_acts) != 1152:
        return sample  # 跳过异常数据

    # === 折叠 1152 → 576 ===
    folded_acts = (selected_acts[:576] + selected_acts[576:]) / 2  # shape = (576,)
    folded_acts = folded_acts.reshape(24, 24)
    H, W = 336, 336
    # === 载入图像并标准化为 RGB ===

    img=image.resize((H, W))
    img = np.array(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # === 图像尺寸与 patch 计算 ===

    patch_count = 24
    patch_size = H // patch_count  # 14

    # === 根据激活值筛选出后 25% patch ===
    threshold = np.percentile(folded_acts, 75)

    for i in range(patch_count):
        for j in range(patch_count):
            if folded_acts[i, j] <= threshold:
                # 将对应 patch 设置为黑色
                img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 0

    # === 更新 image 字段 ===
    new_image = Image.fromarray(img.astype(np.uint8))
    sample["image"] = new_image
    return sample

# 应用到整个数据集
new_data = data.map(process_image_mask, desc="Masking image patches", num_proc=4)

# 保存处理后的数据集
save_path = "/aifs4su/yaodong/changye/data/OKVQA_cosi_Vmasked_0.25"
new_data.save_to_disk(save_path)

print(f"✅ 已保存新数据集至: {save_path}")

import os
import pdb
from typing import Any, cast
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from sae_lens import SAE
from torchvision.transforms.functional import to_pil_image
from transformer_lens.HookedLlava import HookedLlava
from transformer_lens import HookedChameleon
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import ChameleonForConditionalGeneration, AutoTokenizer, ChameleonProcessor
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
) 
# pdb.set_trace()


def load_llava_model(model_name: str, model_path: str, device: str,n_devices:str,stop_at_layer:int=None):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    vision_tower = vision_model.vision_tower.to(device)
    multi_modal_projector = vision_model.multi_modal_projector.to(device)
    hook_language_model = HookedLlava.from_pretrained_no_processing(
        model_name,
        hf_model=vision_model.language_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=n_devices,
        stop_at_layer=stop_at_layer,
    )
    # hook_language_model = None
    del vision_model,vision_tower,multi_modal_projector
    return (
        processor,
        hook_language_model,
    )

def load_chameleon_model(model_name, model_path, device, n_devices, stop_at_layer):
    processor = ChameleonProcessor.from_pretrained(model_path)
    hf_model = ChameleonForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True)
    model = HookedChameleon.from_pretrained(
        model_name,
        hf_model=hf_model,
        device=device,
        n_devices=n_devices,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=processor.tokenizer,
        stop_at_layer=stop_at_layer,
    )
    del hf_model
    return processor, model

def load_sae(sae_path: str, sae_device: str):
    sae = SAE.load_from_pretrained(
        path=sae_path,
        device=sae_device,
    )
    return sae


def load_dataset_func(dataset_path: str, columns_to_read: list):
    try:
        dataset = (
            load_dataset(
                dataset_path,
                split="train",
                streaming=False,
                trust_remote_code=False,  # type: ignore
            )
            if isinstance(dataset_path, str)
            else dataset_path
        )
    except Exception:
        dataset = (
            load_from_disk(
                dataset_path,
            )
            if isinstance(dataset_path, str)
            else dataset_path
        )
    if isinstance(dataset, (Dataset, DatasetDict)):
        dataset = cast(Dataset | DatasetDict, dataset)
    
    if hasattr(dataset, "set_format"):
        dataset.set_format(type="torch", columns=columns_to_read)
        print("Dataset format set.")
    return dataset

def process_single_example(example, system_prompt, user_prompt, assistant_prompt, processor):
    prompt = example['question']
    formatted_prompt = (
        f"{system_prompt}"
        f"{user_prompt.format(input=prompt)}"
        f"{assistant_prompt.format(output='')}"
    )
    image = example['image']
    image = image.resize((336, 336)).convert('RGBA')
    text_input = processor.tokenizer(formatted_prompt, return_tensors="pt")
    image_input = processor.image_processor(images=image, return_tensors="pt")
    return {
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "pixel_values": image_input["pixel_values"],
        "image_sizes": image_input["image_sizes"]
    }

def process_chameleon_single_example(example, system_prompt, user_prompt, assistant_prompt, processor):
    prompt = example['question']
    formatted_prompt = (
        f"{system_prompt}"
        f"{user_prompt.format(input=prompt)}"
        f"{assistant_prompt.format(output='')}"
    )
    image = example['image']
    image = image.resize((336, 336)).convert('RGBA')
    input_ids=processor(image,formatted_prompt).input_ids
    # breakpoint()
    return {"input_ids":input_ids}

def image_recover(inputs, processor):
    img_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    img_recover = inputs.pixel_values[0].cpu() * img_std + img_mean
    img_recover = to_pil_image(img_recover)
    return img_recover

def run_chameleon_model(inputs, hook_language_model, sae, sae_device: str,stop_at_layer):
    with torch.no_grad():
        out, cache = hook_language_model.run_with_cache(
            input=inputs,
            prepend_bos=True,
            names_filter=lambda name: name == sae.cfg.hook_name,
            stop_at_layer=stop_at_layer,
            return_type="gererate_with_saev",
        )
        logit=out[0]
        image_indice=out[1]
        # breakpoint()
        tmp_cache = cache[sae.cfg.hook_name]

        tmp_cache = tmp_cache.to(sae_device)
        feature_acts = sae.encode(tmp_cache)
        # print(feature_acts.shape)
        sae_out = sae.decode(feature_acts)
        del cache
    return tmp_cache,image_indice, feature_acts


def run_model(inputs, hook_language_model, sae, sae_device: str,stop_at_layer):
    with torch.no_grad():
        out, cache = hook_language_model.run_with_cache(
            input=inputs,
            model_inputs=inputs,
            vision=True,
            prepend_bos=True,
            names_filter=lambda name: name == sae.cfg.hook_name,
            stop_at_layer=stop_at_layer,
            return_type="generate_with_saev",
        )
        logit=out[0]
        image_indice=out[1]
        tmp_cache = cache[sae.cfg.hook_name]
        
        tmp_cache = tmp_cache.to(sae_device)
        feature_acts = sae.encode(tmp_cache)
        # print(feature_acts.shape)
        sae_out = sae.decode(feature_acts)
        del cache
    return tmp_cache,image_indice, feature_acts

def cal_chameleon_top_cosimilarity(logits,image_indice, feature_act):
    #选出图片和文本token的top50激活feature(l0)
    # image_indice: Tensor of shape ( num_indices)
    # feature_acts: List or Tensor of shape (sequence_length, feature_dim)
    # logits:(sequence_length, model_dim)
    # breakpoint()
    image_indice=torch.tensor(image_indice).to("cuda")
    logits=logits.to(image_indice.device)
    feature_act=feature_act.to(image_indice.device)
    # assert image_indice.shape[0] == 1176

    values, indices = torch.topk(feature_act, k=50, dim=1)  # values 和 indices 的形状均为 (num_features, 50)
    token_list = []
    for idx, val, logit in zip(indices, values, logits):
        feature_dict = dict(zip(idx.tolist(), val.tolist()))
        token_dict = {
            'features': feature_dict,
            'logits': logit.tolist()  # 将logit张量转换为列表
        }
        token_list.append(token_dict)

    image_token_list = [token_list[i] for i in image_indice.tolist()]

    all_indices = torch.arange(len(token_list))

    text_indices = torch.tensor(list(set(all_indices.tolist()) - set(image_indice.tolist())))

    text_token_list = [token_list[i] for i in text_indices.tolist()]
    return text_token_list, image_token_list

def cal_top_cosimilarity(logits,image_indice, feature_act):
    #选出图片和文本token的top50激活feature(l0)
    # image_indice: Tensor of shape ( num_indices)
    # feature_acts: List or Tensor of shape (sequence_length, feature_dim)
    # logits:(sequence_length, model_dim)
    logits=logits.to(image_indice.device)
    feature_act=feature_act.to(image_indice.device)
    assert image_indice.shape[0] == 1176

    values, indices = torch.topk(feature_act, k=50, dim=1)  # values 和 indices 的形状均为 (num_features, 50)
    token_list = []
    for idx, val, logit in zip(indices, values, logits):
        feature_dict = dict(zip(idx.tolist(), val.tolist()))
        token_dict = {
            'features': feature_dict,
            'logits': logit.tolist()  # 将logit张量转换为列表
        }
        token_list.append(token_dict)

    image_token_list = [token_list[i] for i in image_indice.tolist()]

    all_indices = torch.arange(len(token_list))

    text_indices = torch.tensor(list(set(all_indices.tolist()) - set(image_indice.tolist())))

    text_token_list = [token_list[i] for i in text_indices.tolist()]
    return text_token_list, image_token_list

def separate_feature(image_indice, feature_acts):
    # image_indice: Tensor of shape (batch_size, num_indices)
    # feature_acts: List or Tensor of shape (batch_size, sequence_length, feature_dim)
    batch_size = image_indice.shape[0]
    cooccurrence_features = []
    for i in range(batch_size):
        # For each sample in the batch
        sample_image_indice = image_indice[i]  # shape (num_indices,)
        # Ensure the number of indices matches expected size
        assert sample_image_indice.shape[0] == 1176

        sample_feature_acts = feature_acts[i]  # shape (sequence_length, feature_dim)

        # Convert indices to lists if they are tensors
        sample_image_indice = sample_image_indice.tolist()

        # Separate text and image activations
        text_features_act = torch.cat(
            [sample_feature_acts[:sample_image_indice[0]],
             sample_feature_acts[sample_image_indice[-1]+1:]],
            dim=0
        )
        image_features_act = sample_feature_acts[sample_image_indice[0]: sample_image_indice[-1]+1]

        # Initialize unions
        text_union = set()
        image_union = set()

        # Process text features
        for text_feature in text_features_act:
            text_indices = torch.where(text_feature > 1)[0].tolist()
            text_union.update(text_indices)

        # Process image features
        for image_feature in image_features_act:
            image_indices = torch.where(image_feature > 1)[0].tolist()
            image_union.update(image_indices)

        # Find co-occurrence features
        cooccurrence_feature = list(text_union.intersection(image_union))
        cooccurrence_features.append(cooccurrence_feature)

    return cooccurrence_features


def patch_mapping(image_indice, feature_acts):
    # breakpoint()
    assert image_indice.shape[0] == 1176
    newline_indices = torch.arange(image_indice[0]+576+24-1, image_indice[0]+576*2+24-1, 25)
    valid_indices = torch.tensor(
        [i for i in image_indice if i not in newline_indices]
    )
    # all_indices = torch.arange(feature_acts.size(0))
    # mask = torch.isin(all_indices, image_indice, invert=True)
    # text_features = feature_acts[:,mask]
    patch_indices = torch.stack(
        (valid_indices[:576], valid_indices[576:]), dim=1
    )
    patch_features = feature_acts[:, patch_indices]
    
    patch_features = patch_features.squeeze(0)
    
    activation_l0_norms = (patch_features != 0).sum(dim=2)  # Shape: (576, 2)
    total_activation_l0_norms = activation_l0_norms.to(torch.float32).mean(dim=1)
    # print(patch_features.shape)
    # Step 1: Compute L1 norm over the last dimension (65536) for each activation
    # activation_l1_norms = patch_features.abs().sum(dim=2)  # Shape: (576, 2)

    # # Step 2: Sum the L1 norms over the two activations for each patch
    # total_activation_l1_norms = activation_l1_norms.mean(dim=1)  # Shape: (576,)
    return total_activation_l0_norms,patch_features,feature_acts

def select_patch_mapping(image_indice, feature_acts, selected_feature_indices):
    assert image_indice.shape[0] == 1176
    newline_indices = torch.arange(image_indice[0] + 576 + 24 - 1, image_indice[0] + 576*2 + 24 - 1, 25)
    valid_indices = torch.tensor(
        [i for i in image_indice if i not in newline_indices]
    )

    # 将前 576 个和后 576 个有效 indices 组合成 (576,2) 的 patch 索引
    patch_indices = torch.stack(
        (valid_indices[:576], valid_indices[576:]), dim=1
    )
    
    # 提取对应的 patch 的feature激活值
    # 假设feature_acts的形状是 (1, total_positions, feature_dim)，
    # 其中 total_positions 至少是 image_indice中提取出来的数量对应。
    patch_features = feature_acts[:, patch_indices]   # shape: (1, 576, 2, feature_dim)
    patch_features = patch_features.squeeze(0)        # shape: (576, 2, feature_dim)

    # 根据 selected_feature_indices 对特征维度进行筛选
    # 假设 selected_feature_indices 是一个 1D 的tensor或列表，长度小于等于 feature_dim
    patch_features = patch_features[:, :, selected_feature_indices]  # shape: (576, 2, len(selected_feature_indices))

    # 计算 L0 norm：即统计非零元素的个数
    activation_l0_norms = (patch_features != 0).sum(dim=2)  # shape: (576, 2)
    total_activation_l0_norms = activation_l0_norms.to(torch.float32).mean(dim=1)  # shape: (576,)

    return total_activation_l0_norms, patch_features, feature_acts

def weight_patch_mapping(image_indice, feature_acts, selected_feature_indices):

    assert image_indice.shape[0] == 1176
    newline_indices = torch.arange(
        image_indice[0] + 576 + 24 - 1, 
        image_indice[0] + 576*2 + 24 - 1, 
        25
    )
    valid_indices = torch.tensor([i for i in image_indice if i not in newline_indices])

    # 将前 576 个和后 576 个有效 indices 组合成 (576,2) 的 patch 索引
    patch_indices = torch.stack(
        (valid_indices[:576], valid_indices[576:]), dim=1
    )  # shape: (576, 2)

    # 从 feature_acts 中提取对应 patch 的特征激活值
    # feature_acts.shape: (1, total_positions, feature_dim)
    patch_features = feature_acts[:, patch_indices]   # shape: (1, 576, 2, feature_dim)
    patch_features = patch_features.squeeze(0)        # shape: (576, 2, feature_dim)

    # 将 selected_feature_indices 分解为 indices 和 weights
    # 假设 selected_feature_indices 格式为 [(idx1, w1), (idx2, w2), ...]
    indices = [item[0] for item in selected_feature_indices]
    weights = torch.tensor([item[1] for item in selected_feature_indices], 
                           dtype=patch_features.dtype, device=patch_features.device)
    
    # 使用 indices 对特征维度进行筛选
    # 处理后 patch_features shape: (576, 2, len(indices))
    patch_features = patch_features[:, :, indices]

    # 计算 L0 加权：先得到非零元素的掩码，然后与 weights 对应相乘
    # (patch_features != 0) -> (576, 2, len(indices)) 的布尔张量
    # 转换为浮点型用于数值计算
    nonzero_mask = (patch_features != 0).float()

    # 将非零掩码按相应的特征权重加权
    # weights shape: (len(indices),)，通过广播扩展到 (576, 2, len(indices))
    weighted_nonzero = nonzero_mask * weights

    # 对最后一个维度(特征维度)求和，得到每个patch在两个位置上的加权L0
    activation_l0_norms = weighted_nonzero.sum(dim=2)  # shape: (576, 2)

    # 对两个位置求均值得到最终的加权L0范数 (shape: (576,))
    total_activation_l0_norms = activation_l0_norms.mean(dim=1)

    return total_activation_l0_norms, patch_features, feature_acts


def count_red_blue_elements(activation_colored_uint8, blue_threshold=200, red_threshold=200):
    """
    Counts the number of pixels close to blue and red in an RGB image.
    
    Args:
        activation_colored_uint8 (np.ndarray): RGB image array with shape (H, W, 3).
        blue_threshold (int): Threshold for blue pixel detection.
        red_threshold (int): Threshold for red pixel detection.
    
    Returns:
        int, int: Count of blue-like and red-like pixels.
    """
    blue_mask = (activation_colored_uint8[:, :, 0] < blue_threshold) & \
                (activation_colored_uint8[:, :, 1] < blue_threshold) & \
                (activation_colored_uint8[:, :, 2] > blue_threshold)
    blue_count = blue_mask.sum()
    
    red_mask = (activation_colored_uint8[:, :, 0] > red_threshold) & \
               (activation_colored_uint8[:, :, 1] < red_threshold) & \
               (activation_colored_uint8[:, :, 2] < red_threshold)
    red_count = red_mask.sum()
    
    return blue_count, red_count

def map_patches_to_image(total_activation_l1_norms,lower_clip=0.01,upper_clip=0.99,cmap='plasma',max_val=None):
    """
    Maps activation data from patches to the corresponding positions in the image.

    Args:
        patch_features (torch.Tensor): Activation data of shape (576, 2, 65536).

    Returns:
        Image: A PIL Image representing the activation map.
    """

    
    lower_bound = torch.quantile(total_activation_l1_norms, lower_clip)
    upper_bound = torch.quantile(total_activation_l1_norms, upper_clip)
    
    clipped_activation_l1_norms = torch.where(total_activation_l1_norms < lower_bound, 
                                          torch.tensor(0.0, device=total_activation_l1_norms.device), 
                                          total_activation_l1_norms)
    clipped_activation_l1_norms = torch.where(clipped_activation_l1_norms > upper_bound, 
                                          upper_bound, 
                                          clipped_activation_l1_norms)
    # print("After clipping:")
    # print(f"  Elements equal to 0 (near lower bound): {(clipped_activation_l1_norms == 0).sum().item()}")
    # print(f"  Elements equal to upper bound: {(clipped_activation_l1_norms == upper_bound).sum().item()}")

    # Step 3: Reshape total_activation_l1_norms into a 24x24 grid
    activation_l1_norms_2d = clipped_activation_l1_norms.view(24, 24)

    # Step 4: Upsample the 24x24 grid to a 336x336 image by repeating each element into a 14x14 block
    activation_l1_norms_large = activation_l1_norms_2d.repeat_interleave(14, dim=0).repeat_interleave(14, dim=1)

    # Step 5: Normalize activation_l1_norms to [0, 1] for image representation
    activation_l1_norms_large = activation_l1_norms_large.float()

    

    # 计算绝对值的最大值
    if max_val is None:
        max_abs_val = activation_l1_norms_large.abs().max()
    else:
        max_abs_val=max_val

    # 避免除以零的情况
    if max_abs_val == 0:
        print("max_abs_val == 0")
        activation_l1_norms_normalized = torch.zeros_like(activation_l1_norms_large)
    else:
        # 归一化到 [-1, 1]
        activation_l1_norms_normalized = activation_l1_norms_large / max_abs_val
        if activation_l1_norms_normalized.min()<0:
            # 平移到 [0, 1]
            activation_l1_norms_normalized = (activation_l1_norms_normalized + 1) / 2
    
    # print("After normalization:")
    # print(f"  Min value: {activation_l1_norms_normalized.min().item()}")
    # print(f"  Max value: {activation_l1_norms_normalized.max().item()}")
    # print(f"  Elements close to -1 (blue): {(activation_l1_norms_normalized < -0.3).sum().item()}")
    # print(f"  Elements close to 1 (red): {(activation_l1_norms_normalized > 0.3).sum().item()}")
    
    # Step 6: Apply a different colormap for a heatmap style
    colormap = plt.get_cmap(cmap)  # Try 'plasma', 'inferno', or 'magma' for similar effects
    # if activation_l1_norms_normalized.min()<0:
    #     # print("Detected negative values, adjusting colormap normalization.")
    #     activation_colored = colormap((activation_l1_norms_normalized.cpu().numpy() + 1) / 2) 
    # else:
    activation_colored = colormap(activation_l1_norms_normalized.cpu().numpy())

    # Step 7: Convert to NumPy array and ensure data type is uint8
    activation_colored_uint8 = (activation_colored[:, :, :3] * 255).astype(np.uint8)
    # print(f"  Activation map shape: {activation_colored_uint8.shape}")
    # blue_count, red_count = count_red_blue_elements(activation_colored_uint8)
    # print(f"Number of blue-like elements: {blue_count}")
    # print(f"Number of red-like elements: {red_count}")
    # Step 8: Create an RGB image from the array
    activation_map = Image.fromarray(activation_colored_uint8)

    return activation_map


def overlay_activation_on_image(image, activation_map,alpha=128):
    original_image = image.convert('RGBA')
    activation_map = activation_map.resize((336, 336)).convert('RGBA')

    # Adjust the transparency of the activation map
    # alpha = 128  # 0.5 transparency, value range 0-255
    activation_map.putalpha(alpha)

    # Overlay the activation map on the original image
    combined = Image.alpha_composite(original_image, activation_map)

    return combined


def filter_diff_by_std(diff):
    """
    Filters the diff tensor to retain only the values beyond one standard deviation from the mean.

    Args:
        diff (torch.Tensor): The difference tensor containing positive and negative shifts.

    Returns:
        torch.Tensor: A tensor with values within one standard deviation set to 0.
    """
    # 计算 diff 的均值和标准差
    mean_diff = diff.mean()
    std_diff = diff.std()

    # 定义上下阈值
    upper_threshold = mean_diff + std_diff
    lower_threshold = mean_diff - std_diff

    # 过滤掉在一个标准差以内的值
    filtered_diff = torch.where(
        (diff > upper_threshold) | (diff < lower_threshold),
        diff,
        torch.tensor(0.0, device=diff.device)
    )

    return filtered_diff

def generate_with_saev(inputs, hook_language_model, processor, save_path, image, sae, sae_device: str,max_new_tokens=100,selected_feature_indices=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        tokens, image_indice, tmp_cache_list = hook_language_model.generate(
            inputs,
            sae_hook_name=sae.cfg.hook_name,
            max_new_tokens=max_new_tokens,
        )
        input_length = inputs['input_ids'].shape[1]
        decoded_tokens = processor.tokenizer.convert_ids_to_tokens(tokens[0])
        output = processor.decode(tokens[0], skip_special_tokens=True)
        previous_total_activation_l1_norms = None
        previous_patch_features=None

        # print(output)
        
        tmp_cache_list=[tmp_cache.to(sae_device) for tmp_cache in tmp_cache_list]
        feature_acts_list = [sae.encode(tmp_cache) for tmp_cache in tmp_cache_list]
        image_indice = image_indice.to("cpu")
        image_indice=image_indice.squeeze(0)
        feature_acts_list = [feature_acts.to("cpu") for feature_acts in feature_acts_list]
        # print(feature_acts_list)
        if selected_feature_indices is None:
            total_activation_l0_norms_list,patch_features_list,feature_acts_list = zip(*[patch_mapping(image_indice, feature_acts) for feature_acts in feature_acts_list])
        elif selected_feature_indices is not None and type(selected_feature_indices[0])==tuple:
            total_activation_l0_norms_list,patch_features_list,feature_acts_list = zip(*[weight_patch_mapping(image_indice, feature_acts, selected_feature_indices) for feature_acts in feature_acts_list])
        else:
            total_activation_l0_norms_list,patch_features_list,feature_acts_list = zip(*[select_patch_mapping(image_indice, feature_acts, selected_feature_indices) for feature_acts in feature_acts_list])
        # current_activation_map = map_patches_to_image(total_activation_l1_norms_list[0])
        # final_image = overlay_activation_on_image(image, current_activation_map)
        # final_image.save(os.path.join(save_path, f"current.png"))
        # upper_bound = torch.quantile(torch.cat(total_activation_l1_norms_list), 0.99)
        
        
        # for i, (tmp_cache, token) in enumerate(zip(tmp_cache_list, decoded_tokens[input_length:])):
        #     tmp_cache = tmp_cache.to(sae_device)
        #     feature_acts = sae.encode(tmp_cache)
        #     image_indice = image_indice.to("cpu")
        #     feature_acts = feature_acts.to("cpu")
        #     total_activation_l1_norms,patch_features = patch_mapping(image_indice, feature_acts)
        #     patch_features=patch_features[(patch_features!=0).sum(dim=2)>50]
        #     if previous_patch_features is not None:
        #         patch_features=patch_features.to("cuda")
        #         previous_patch_features=previous_patch_features.to("cuda")
        #         cos_sim = F.cosine_similarity(patch_features, previous_patch_features, dim=1)
        #         mean_cos_sim = cos_sim.mean()
        #         print(f"mean_cos_sim:{i}_{token}",mean_cos_sim)
            # print(total_activation_l1_norms.sum())
            # if previous_total_activation_l1_norms is None: 
                # current_activation_map = map_patches_to_image(total_activation_l1_norms)
                # final_image = overlay_activation_on_image(image, current_activation_map)
                # final_image.save(os.path.join(save_path, f"current_{i}_{token}.png"))
            # current_activation_map = map_patches_to_image(total_activation_l1_norms)
            # final_image = overlay_activation_on_image(image, current_activation_map)
            # final_image.save(os.path.join(save_path, f"current_{i}_{token}.png"))
            # else:
            #     # patch_features=patch_features.to("cuda")
            #     # previous_patch_features=previous_patch_features.to("cuda")
            #     # cos_sim = F.cosine_similarity(patch_features, previous_patch_features, dim=1)
            #     # print(f"total_activation_l1_norms:",total_activation_l1_norms.sum())
            #     # print(f"previous_total_activation_l1_norms:",previous_total_activation_l1_norms.sum())
            #     diff = total_activation_l1_norms - previous_total_activation_l1_norms
            #     shift_diff = filter_diff_by_std(diff)  # Shift to non-negative range
            #     diff_activation_map = map_patches_to_image(shift_diff, lower_clip=0.01, upper_clip=0.99, cmap='bwr')
            #     final_image = overlay_activation_on_image(image, diff_activation_map)
            #     final_image.save(os.path.join(save_path, f"diff_{i}_{token}.png"))
            #     if i%10==0:
            #         current_activation_map=map_patches_to_image(total_activation_l1_norms)
            #         final_image = overlay_activation_on_image(image, current_activation_map)
            #         final_image.save(os.path.join(save_path, f"current_{i}_{token}.png"))

            # previous_total_activation_l1_norms = total_activation_l1_norms
            # previous_patch_features=patch_features
        # tmp_cache = cache[sae.cfg.hook_name]
        # tmp_cache = tmp_cache.to(sae_device)
        # feature_acts = sae.encode(tmp_cache)
        # sae_out = sae.decode(feature_acts)
        # del cache
        # plt.figure(figsize=(10, 6))
        # plt.plot(activation_diff, marker='o', linestyle='-', color='b')
        # plt.xlabel("Generation Step")
        # plt.ylabel("activation_diff")
        # plt.title("activation_diff Value over Generation Steps")
        # plt.grid(True)
        # plt.savefig(os.path.join(save_path, "activation_diff_plot.png"))
        # plt.show()
    return total_activation_l0_norms_list,patch_features_list,feature_acts_list,image_indice,output


def main():
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_path = "/mnt/data/changye/model/llava"
    device = "cuda:5"
    sae_device = "cuda:6"
    sae_path = "/mnt/data/changye/checkpoints/checkpoints-V/kxpk98cr/final_122880000"
    dataset_path = "/mnt/data/changye/data/obelics3k-tokenized-llava4096"
    columns_to_read = ["input_ids", "pixel_values", "attention_mask", "image_sizes"]
    example_prompt = "What is shown in this image?"
    # example_answer = "Apple"
    image_path = "/home/saev/changye/blue.jpg"
    save_path = "/home/saev/changye/SAELens-V/activation_visualization"


    (
        processor,
        vision_model,
        vision_tower,
        multi_modal_projector,
        hook_language_model,
    ) = load_llava_model(MODEL_NAME, model_path, device)

    sae = load_sae(sae_path, sae_device)

    # dataset = load_dataset_func(dataset_path, columns_to_read)

    inputs, image = prepare_input(processor,device, image_path, example_prompt)
    _ = generate_with_saev(
        inputs, hook_language_model, processor, save_path, image, sae, sae_device
    )

    # img_recover = image_recover(inputs, processor)

    # image_indice, feature_acts = run_model(inputs, hook_language_model, sae, sae_device)

    # image_indice = image_indice.to("cpu")
    # feature_acts = feature_acts.to("cpu")

    # patch_features = patch_mapping(image_indice, feature_acts)

    # activation_map = map_patches_to_image(patch_features)

    # final_image = overlay_activation_on_image(image, activation_map)
    # final_image.show()
    # final_image.save("car.png")


if __name__ == "__main__":
    main()

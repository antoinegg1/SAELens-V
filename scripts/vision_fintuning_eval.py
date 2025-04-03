import argparse
import json
import threading
import tqdm
import torch
from typing import Tuple
from functools import partial
import os
# ============== 这里引入你已有的模块或第三方库 ==============
from datasets import load_dataset, load_from_disk
from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformer_lens.HookedLlava import HookedLlava
from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import utils

# ============== 你已有的函数：reconstruction hook等 ==============
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

# ============== 文本重建测试 ==============
def text_reconstruction_test(hook_language_model, sae, token_dataset, description, batch_size=8):
    total_orig_loss = 0.0
    total_reconstr_loss = 0.0
    total_zero_loss = 0.0
    num_batches = 0

    total_tokens = 50
    with torch.no_grad():
        with tqdm.tqdm(total=total_tokens, desc="Text Reconstruction Test") as pbar:
            for start_idx in range(0, total_tokens, batch_size):
                end_idx = min(start_idx + batch_size, total_tokens)
                batch_data = token_dataset[start_idx:end_idx]

                # 获取批量的 tokens
                batch_tokens = batch_data["tokens"]

                # 确保 tokens 在模型的设备上
                device = hook_language_model.cfg.device
                batch_tokens = batch_tokens.to(device)

                # 获取模型的激活并缓存
                _, cache = hook_language_model.run_with_cache(
                    batch_tokens,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )

                # 使用 SAE 进行编码和解码
                activation = cache[sae.cfg.hook_name]
                activation = activation.to(sae.device)
                feature_acts = sae.encode(activation)
                sae_out = sae.decode(feature_acts)
                sae_out = sae_out.to(device)
                del cache  # 释放缓存

                # 计算原始损失
                orig_loss = hook_language_model(
                    batch_tokens, return_type="loss"
                ).item()
                total_orig_loss += orig_loss

                # 计算重建后的损失
                reconstr_loss = hook_language_model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[(sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_out))],
                    return_type="loss",
                ).item()
                total_reconstr_loss += reconstr_loss

                # 计算零置换后的损失
                zero_loss = hook_language_model.run_with_hooks(
                    batch_tokens,
                    fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
                    return_type="loss",
                ).item()
                total_zero_loss += zero_loss

                num_batches += 1

                # 清理中间变量
                del activation, feature_acts, sae_out
                torch.cuda.empty_cache()

                pbar.update(batch_size)

    avg_orig_loss = total_orig_loss / num_batches
    avg_reconstr_loss = total_reconstr_loss / num_batches
    avg_zero_loss = total_zero_loss / num_batches

    print(f"{description}_平均原始损失:", avg_orig_loss)
    print(f"{description}_平均重建损失:", avg_reconstr_loss)
    print(f"{description}_平均零置换损失:", avg_zero_loss)


# ============== 图像重建测试 ==============
def image_reconstruction_test(hook_language_model, sae, token_dataset, description, batch_size=8):
    total_orig_loss = 0.0
    total_reconstr_loss = 0.0
    total_zero_loss = 0.0
    num_batches = 0
    total_tokens = 50
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(total=total_tokens, desc="Image Reconstruction Test") as pbar:
            for data in token_dataset:
                count += 1
                if count > total_tokens:
                    break

                device = hook_language_model.cfg.device
                token ={
                    "input_ids": torch.tensor(data["input_ids"]).to(device),
                    "pixel_values": torch.tensor(data["pixel_values"]).to(device),
                    "attention_mask": torch.tensor(data["attention_mask"]).to(device),
                    "image_sizes": torch.tensor(data["image_sizes"]).to(device)
                }

                # 获取模型的激活并缓存
                _, cache = hook_language_model.run_with_cache(
                    input=token,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )

                # 使用 SAE 进行编码和解码
                activation = cache[sae.cfg.hook_name]
                activation = activation.to(sae.device)
                feature_acts = sae.encode(activation)
                sae_out = sae.decode(feature_acts)
                sae_out = sae_out.to(device)
                del cache  # 释放缓存

                # 计算原始损失
                orig_loss = hook_language_model(token, return_type="loss").item()
                total_orig_loss += orig_loss

                # 计算重建后的损失
                reconstr_loss = hook_language_model.run_with_hooks(
                    token,
                    fwd_hooks=[(sae.cfg.hook_name, partial(reconstr_hook, sae_out=sae_out))],
                    return_type="loss",
                ).item()
                total_reconstr_loss += reconstr_loss

                # 计算零置换后的损失
                zero_loss = hook_language_model.run_with_hooks(
                    token,
                    fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
                    return_type="loss",
                ).item()
                total_zero_loss += zero_loss

                num_batches += 1

                del activation, feature_acts, sae_out
                torch.cuda.empty_cache()
                pbar.update(1)

    avg_orig_loss = total_orig_loss / num_batches
    avg_reconstr_loss = total_reconstr_loss / num_batches
    avg_zero_loss = total_zero_loss / num_batches

    print(f"{description}_平均原始损失:", avg_orig_loss)
    print(f"{description}_平均重建损失:", avg_reconstr_loss)
    print(f"{description}_平均零置换损失:", avg_zero_loss)


# ============== L0 测试（演示的是图像数据结构） ==============
def l0_test(sae, hook_language_model, token_dataset, description, batch_size=8):
    sae.eval() 
    total_tokens = 200
    tok = 0
    l0_list = []

    with torch.no_grad():
        with tqdm.tqdm(total=total_tokens, desc="L0 Test") as pbar:
            for data in token_dataset:
                tok += 1
                if tok > total_tokens:
                    break

                device = hook_language_model.cfg.device
                tokens = {
                    "input_ids": torch.tensor(data["input_ids"]).to(device),
                    "pixel_values": torch.tensor(data["pixel_values"]).to(device),
                    "attention_mask": torch.tensor(data["attention_mask"]).to(device),
                    "image_sizes": torch.tensor(data["image_sizes"]).to(device)
                }

                # 获取缓存
                _, cache = hook_language_model.run_with_cache(
                    tokens,
                    prepend_bos=True,
                    names_filter=lambda name: name == sae.cfg.hook_name
                )

                tmp_cache = cache[sae.cfg.hook_name].to(sae.device)
                del cache

                # SAE 编码解码
                feature_acts = sae.encode(tmp_cache)
                sae.decode(feature_acts)  # 这里 decode 的结果没用到，但可按需使用

                # 计算 L0 范数
                l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
                result_list = l0.flatten().tolist()
                l0_list.extend(result_list)

                torch.cuda.empty_cache()
                pbar.update(1)

        l0_average = sum(l0_list) / len(l0_list)
        print(f"平均 L0 for {description}: {l0_average}")


# ============== 加载模型相关 ==============
def load_vision_model(model_path: str, device: str) -> Tuple[LlavaNextForConditionalGeneration, torch.nn.Module, torch.nn.Module]:
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    vision_tower = vision_model.vision_tower.to(device)
    multi_modal_projector = vision_model.multi_modal_projector.to(device)
    return vision_model, vision_tower, multi_modal_projector


def load_hooked_llava(model_name: str, hf_model, device: str, vision_tower, multi_modal_projector, n_devices: int) -> HookedLlava:
    hook_language_model = HookedLlava.from_pretrained(
        model_name,
        hf_model=hf_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=None,
        dtype=torch.float32,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        n_devices=n_devices,
    )
    return hook_language_model

def load_sae_model(path: str, device: str) -> SAE:
    sae_model = SAE.load_from_pretrained(
        path=path,
        device=device
    )
    return sae_model


# ============== 加载&标记化数据集（主要用于文本） ==============
def load_and_tokenize_dataset(dataset_path: str, split: str, tokenizer, max_length: int, add_bos_token: bool):
    dataset = load_dataset(
        path=dataset_path,
        split=split,
        streaming=False,
    )
    tokenized_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=True,
        max_length=max_length,
        add_bos_token=add_bos_token,
    )
    return tokenized_dataset


# ============== 统一封装的测试函数调用 ==============
def run_l0_test(sae, hook_model, token_dataset, description, batch_size=8):
    print(f"[Run L0 Test] {description}")
    l0_test(sae, hook_model, token_dataset, description, batch_size)


def run_reconstruction_test(hook_language_model, sae, token_dataset, description, batch_size=8, is_image=False):
    print(f"[Run Reconstruction Test] {description}, is_image={is_image}")
    if is_image:
        image_reconstruction_test(hook_language_model, sae, token_dataset, description, batch_size)
    else:
        text_reconstruction_test(hook_language_model, sae, token_dataset, description, batch_size)


# ============== Argparse & Main ==============
def parse_args():
    parser = argparse.ArgumentParser(description="Unified testing script for reconstruction and L0.")
    parser.add_argument("--model_path", type=str, default="/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-vicuna-13b-hf",
                        help="The path to the LLAVA or Chameleon model (local or HF Hub).")
    parser.add_argument("--sae_path", type=str, default="/aifs4su/yaodong/changye/checkpointsV-V13-pile10k/cy1jfqit/final_30720000",
                        help="The path to the trained SAE model.")
    parser.add_argument("--data_path", type=str, default="/aifs4su/yaodong/changye/data/Antoinegg1/obelics_100k-tokenized-4image_llava_vicuna-13B_4096/batch_1",help="Path to either a local dataset folder or JSON file.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to use, e.g., cuda:0.")
    parser.add_argument("--n_devices", type=int, default=8, help="Number of devices for Hooked model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing.")
    parser.add_argument("--is_image", default=True,
                        help="If set, use image reconstruction logic; otherwise text-based.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-vicuna-13b-hf",
                        help="Model name string for hooking if needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    # 1. 加载模型（根据 model_path 判断是 LLAVA 还是 Chameleon）
    if "llava" in args.model_path.lower():
        vision_model, vision_tower, multi_modal_projector = load_vision_model(
            model_path=args.model_path,
            device=device
        )
        # LLAVA 的语言部分
        hf_language_model = vision_model.language_model
        # HookedLlava
        hook_model = load_hooked_llava(
            model_name=args.model_name,
            hf_model=hf_language_model,
            device=device,
            vision_tower=vision_tower,
            multi_modal_projector=multi_modal_projector,
            n_devices=args.n_devices,
        )
    else:
        raise ValueError("Currently only supports 'llava' in model_path.")

    # 2. 加载 SAE
    sae_model = load_sae_model(path=args.sae_path, device=device)

    # 3. 加载数据（从 dataset 或从 JSON）
    if os.path.isdir(args.data_path):
        # 如果是文本数据集，需要在 load_and_tokenize_dataset 里进行 tokenize。
        # 这里示例仅调用 load_from_disk，实际需要结合 tokenize 而定。
        print(f"Loading dataset from disk: {args.data_path}")
        tokenized_dataset = load_from_disk(args.data_path)
    else:
        # 从本地 JSON 文件中加载
        print(f"Loading dataset from JSON file: {args.data_path}")
        with open(args.data_path, "r", encoding="utf-8") as f:
            tokenized_dataset = json.load(f)

    run_reconstruction_test(
        hook_language_model=hook_model,
        sae=sae_model,
        token_dataset=tokenized_dataset,
        description="SAE_reconstruction_Test",
        batch_size=args.batch_size,
        is_image=args.is_image
    )
    run_l0_test(
        sae=sae_model,
        hook_model=hook_model,
        token_dataset=tokenized_dataset,
        description="SAE_l0_Test",
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

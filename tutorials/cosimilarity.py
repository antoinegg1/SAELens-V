#!/usr/bin/env python
# coding: utf-8
'''
python /mnt/file2/changye/SAELens-V/tutorials/cosimilarity.py \
  --model_name "llava-hf/llava-v1.6-mistral-7b-hf" \
  --model_path "/mnt/file2/changye/dataset/htlou/mm-interp-AA_preference_cocour_new_step10_0_100-llava-mistral" \
  --sae_path "/mnt/file2/changye/model/SAE/llavasae_obliec100k_SAEV" \
  --sae_device "cuda:4" \
  --device "cuda:5" \
  --dataset_path "/mnt/file2/changye/dataset/RLAIF-V-Dataset1k" \
  --system_prompt " " \
  --user_prompt "USER: \n<image> {input}" \
  --assistant_prompt "\nASSISTANT: {output}" \
  --output_dir "/mnt/file2/changye/dataset/interp/Align-Anything-preference_llava_interp/AA_preference_cocour_new_step10_0_100_cosi_weight" \
  --num_proc 8 \
  --n_devices 4 \
  --stop_at_layer 17 \
  --feature_num 65536 \
  --top_k 30 \
  --text_image_top_k 5
'''
import os
import argparse
import torch
import tqdm
import plotly.express as px
import random
from datasets import load_from_disk
from transformer_lens import HookedTransformer
from typing import Any, Generator, Iterator, Literal, cast
from sae_lens import SAE
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)
from PIL import Image
import io
import numpy as np
import concurrent.futures
from transformer_lens.HookedLlava import HookedLlava
from sae_lens.activation_visualization import (
    load_llava_model,
    load_sae,
    separate_feature,
    run_model,
    cal_top_cosimilarity,
    process_single_example,
)
from functools import partial

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--model_path", type=str, default="/mnt/file2/changye/model/llava_AA_preference_cocour_0_25")
    parser.add_argument("--sae_path", type=str, default="/mnt/file2/changye/model/llavasae_obliec100k_SAEV")
    parser.add_argument("--sae_device", type=str, default="cuda:7")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_path", type=str, default="/mnt/file2/changye/dataset/Align-Anything_preference_1k")
    parser.add_argument("--system_prompt", type=str, default=" ")
    parser.add_argument("--user_prompt", type=str, default='USER: \n<image> {input}')
    parser.add_argument("--assistant_prompt", type=str, default='\nASSISTANT: {output}')
    parser.add_argument("--split_token", type=str, default='ASSISTANT:')
    parser.add_argument("--output_dir", type=str, default="/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_cooccur_0_25")
    parser.add_argument("--num_proc", type=int, default=60)
    parser.add_argument("--n_devices", type=int, default=4)
    parser.add_argument("--stop_at_layer", type=int, default=17)
    parser.add_argument("--feature_num", type=int, default=65536)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--text_image_top_k", type=int, default=5)
    return parser.parse_args()


def prepare_data(args, processor):
    eval_dataset = load_from_disk(args.dataset_path)
    if "compcap" in args.dataset_path.lower():
        def process_compcap(example):
            # 读取并处理图片
            image = Image.open(io.BytesIO(example['image']))
            example['Image'] = image
            example['prompt']=example['conversations'][0]['value'].replace('<image>',' ') 
            return example

        # 修改或添加新的特征
        eval_dataset = eval_dataset.map(process_compcap)
    # breakpoint()
    process_fn = partial(
        process_single_example,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        assistant_prompt=args.assistant_prompt,
        processor=processor
    )
    inputs = eval_dataset.map(
        process_fn,
        num_proc=args.num_proc,
        batched=False,
        desc="Processing dataset",
        remove_columns=eval_dataset.column_names,
    )
    return inputs

def compute_cosim_and_collect_tokens(inputs, hook_language_model, sae, args):
    text_token_meta_list = []
    image_token_meta_list = []
    with tqdm.tqdm(total=len(inputs), desc="Processing cosimilarity") as pbar:
        for input_data in inputs:
            input_ids = torch.tensor(input_data["input_ids"]).clone().to(args.device)
            attention_mask = torch.tensor(input_data["attention_mask"]).clone().to(args.device)
            pixel_values = torch.tensor(input_data["pixel_values"]).clone().to(args.device)
            image_sizes = torch.tensor(input_data["image_sizes"]).clone().to(args.device)

            processed_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            }
            tmp_cache, image_indices, feature_act = run_model(
                processed_input, 
                hook_language_model, 
                sae, 
                args.sae_device, 
                stop_at_layer=args.stop_at_layer
            )
            text_token_list, image_token_list = cal_top_cosimilarity(
                tmp_cache[0], 
                image_indices[0], 
                feature_act[0]
            )
            text_token_meta_list.append(text_token_list)
            image_token_meta_list.append(image_token_list)
            pbar.update(1)
    return text_token_meta_list, image_token_meta_list

def process_activations_and_save(text_token_meta_list, image_token_meta_list, args):
    flattened_text_list = np.concatenate(text_token_meta_list).tolist()
    flattened_image_list = np.concatenate(image_token_meta_list).tolist()
    feature_num = args.feature_num
    features_top = [[] for _ in range(feature_num)]

    with tqdm.tqdm(total=len(flattened_text_list), desc="Processing flattened_text_list") as pbar:
        for token in flattened_text_list:
            for feature_index, activation_value in token['features'].items():
                features_top[feature_index].append((activation_value, token, 'text'))
            pbar.update(1)
            pbar.refresh()

    with tqdm.tqdm(total=len(flattened_image_list), desc="Processing flattened_image_list") as pbar:
        for token in flattened_image_list:
            for feature_index, activation_value in token['features'].items():
                features_top[feature_index].append((activation_value, token, 'image'))
            pbar.update(1)
            pbar.refresh()

    with tqdm.tqdm(total=feature_num, desc="Processing feature_num") as pbar:
        for i in range(feature_num):
            tokens_with_activation = features_top[i]
            if tokens_with_activation:
                tokens_with_activation.sort(key=lambda x: x[0], reverse=True)
                features_top[i] = tokens_with_activation[: args.top_k]
            else:
                features_top[i] = []
            pbar.update(1)
            pbar.refresh()

    text_feature_list = []
    image_feature_list = []
    cosi_feature_list = []

    with tqdm.tqdm(total=feature_num, desc="Processing feature_num") as pbar:
        for i in range(feature_num):
            tokens = features_top[i]
            if not tokens:
                pbar.update(1)
                continue
            text_tokens = []
            image_tokens = []
            for activation_value, token, token_type in tokens:
                if token_type == 'text':
                    text_tokens.append((activation_value, token))
                elif token_type == 'image':
                    image_tokens.append((activation_value, token))
            if len(text_tokens) == len(tokens):
                text_feature_list.append(i)
            elif len(image_tokens) == len(tokens):
                image_feature_list.append(i)
            else:
                if len(text_tokens) >= args.text_image_top_k and len(image_tokens) >= args.text_image_top_k:
                    top_text_tokens = text_tokens[: args.text_image_top_k]
                    top_image_tokens = image_tokens[: args.text_image_top_k]
                    text_logits = [tok['logits'] for _, tok in top_text_tokens]
                    image_logits = [tok['logits'] for _, tok in top_image_tokens]
                    cosine_similarities = []
                    for t_logit in text_logits:
                        for i_logit in image_logits:
                            t_logit = np.array(t_logit)
                            i_logit = np.array(i_logit)
                            numerator = np.dot(t_logit, i_logit)
                            denominator = np.linalg.norm(t_logit) * np.linalg.norm(i_logit)
                            if denominator == 0:
                                cosine_similarity = 0
                            else:
                                cosine_similarity = numerator / denominator
                            cosine_similarities.append(cosine_similarity)
                    average_cosine_similarity = np.mean(cosine_similarities)
                    cosi_feature_list.append((i, average_cosine_similarity))
            pbar.update(1)
            pbar.refresh()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 写入 text_feature_list
    text_feature_file = os.path.join(args.output_dir, 'text_feature_list.txt')
    with open(text_feature_file, 'w') as f:
        for feature_index in text_feature_list:
            f.write(f"{feature_index}\n")

    # 写入 image_feature_list
    image_feature_file = os.path.join(args.output_dir, 'image_feature_list.txt')
    with open(image_feature_file, 'w') as f:
        for feature_index in image_feature_list:
            f.write(f"{feature_index}\n")

    # 写入 cosi_feature_list
    cosi_feature_file = os.path.join(args.output_dir, 'cosi_feature_list.txt')
    with open(cosi_feature_file, 'w') as f:
        for feature_index, average_cosine_similarity in cosi_feature_list:
            f.write(f"{feature_index},{average_cosine_similarity}\n")

def main():
    args = parse_arguments()
    os.environ["http_proxy"] = "http://127.0.0.1:7895"
    os.environ["https_proxy"] = "http://127.0.0.1:7895"

    processor, hook_language_model = load_llava_model(
        args.model_name, 
        args.model_path, 
        args.device,
        n_devices=args.n_devices, 
        stop_at_layer=args.stop_at_layer
    )
    sae = load_sae(args.sae_path, args.sae_device)

    inputs = prepare_data(args, processor)
    text_token_meta_list, image_token_meta_list = compute_cosim_and_collect_tokens(
        inputs, 
        hook_language_model, 
        sae, 
        args
    )
    process_activations_and_save(text_token_meta_list, image_token_meta_list, args)

if __name__ == "__main__":
    main()
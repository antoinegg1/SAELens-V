import argparse
import os
import tqdm
import hashlib
import logging

from vllm import LLM, SamplingParams
from transformers import LlavaNextProcessor
from PIL import Image
from datasets import load_from_disk

logging.basicConfig(level=logging.WARNING)
'''
python /aifs4su/yaodong/changye/SAELens-V/tutorials/VQA_test.py \
  --data_set /aifs4su/yaodong/changye/data/OKVQA_cosi_Tmasked_0.5 \
  --save_path /aifs4su/yaodong/changye/results/OKVQA_eval_Tmasked_0.5_p \
  --batch_size 8

'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dataset_path = args.data_set
    save_path = args.save_path
    batch_size = args.batch_size

    model_path = "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=8)

    # === 加载数据集 ===
    dataset = load_from_disk(dataset_path)

    # === 图像预处理 ===
    def preprocess_image(image):
        try:
            return image.resize((336, 336)).convert("RGB")
        except:
            return None

    dataset = dataset.map(lambda x: {"image": preprocess_image(x["image"])})
    dataset = dataset.filter(lambda x: x["image"] is not None)

    # === 批量请求构造：直接用已有 prompt + image ===
    def prepare_requests(batch):
        return [
            {
                "prompt": f"{prompt}",
                "multi_modal_data": {"image": image},
            }
            for prompt, image in zip(batch["prompt"], batch["image"])
        ]

    # === 推理参数（严格控制输出） ===
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        stop=["\n"]
    )

    # === 推理主循环 ===
    num_samples = len(dataset)
    predictions = [None] * num_samples
    correct = 0

    for i in tqdm.trange(0, num_samples, batch_size):
        batch = dataset[i: i + batch_size]
        requests = prepare_requests(batch)
        outputs = llm.generate(requests, sampling_params=sampling_params)

        for j, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            global_idx = i + j
            gt_idx = batch["correct_choice_idx"][j]

            # 提取数字（1-based → 0-based）
            try:
                pred_idx = int(text) - 1
            except:
                pred_idx = -1  # 无法解析视为错误

            predictions[global_idx] = pred_idx
            if pred_idx == gt_idx:
                correct += 1

            print(f"[{global_idx}] Pred: {pred_idx}, GT: {gt_idx}, Raw: '{text}'")

    # === 保存结果 ===
    dataset = dataset.add_column("predicted_choice_idx", predictions)
    accuracy = correct / num_samples
    print(f"\n✅ 推理完成，共 {num_samples} 条，准确率：{accuracy:.2%}")

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "accuracy.txt"), "w") as f:
        f.write(f"{accuracy:.4f}\n")

    dataset.save_to_disk(save_path)
    print(f"📁 已保存预测结果至：{save_path}")

if __name__ == "__main__":
    main()

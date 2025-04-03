import argparse
import os
import tqdm
import hashlib
import logging

from vllm import LLM, SamplingParams
from transformers import LlavaNextProcessor
from PIL import Image
from datasets import load_from_disk

logging.basicConfig(level=logging.WARNING)  # 只显示 WARNING 及以上级别的日志


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run LLaVA classification and save results as Hugging Face Dataset.")
    parser.add_argument("--data_set",default="/aifs4su/yaodong/changye/data/Antoinegg1/Semantic_data/imagenet10_attn_25", type=str, help="Path to the local dataset to load using load_from_disk")
    parser.add_argument("--save_path",default="/aifs4su/yaodong/changye/data/semantic_result/imagenet10_attn_25_eval", type=str, help="Path to save the new huggingface dataset (save_to_disk)")
    args = parser.parse_args()

    dataset_path = args.data_set
    save_path = args.save_path

    # 模型路径（请根据实际情况修改）
    llava_model_path = "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf"
    
    # 加载本地数据集
    local_dataset = load_from_disk(dataset_path)

    # 定义图像预处理函数
    def preprocess_image(image):
        try:
            image = image.resize((336, 336)).convert('RGB')
            final_image = image
        except Exception as e:
            final_image = None
            print(f"Image preprocessing failed: {e}")
        return final_image

    # 对数据集中的图像进行预处理
    local_dataset = local_dataset.map(lambda x: {"image": preprocess_image(x["image"])})

    # 过滤掉无效的数据（图像预处理失败或返回 None）
    local_dataset = local_dataset.filter(lambda x: x["image"] is not None)

    # 创建一个保存图片的文件夹（如需）
    images_save_dir = "saved_images"
    os.makedirs(images_save_dir, exist_ok=True)

    # 初始化处理器与 LLM
    processor = LlavaNextProcessor.from_pretrained(llava_model_path)
    llm = LLM(model=llava_model_path, tensor_parallel_size=4)  # 根据硬件条件调整

    # 提示语
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

    # 定义批处理输入的函数
    def prepare_inputs_batch(prompt, images):
        request_list = []
        for img in images:
            request_list.append(
                {
                    "prompt": f"USER: <image>{prompt}\nASSISTANT:",
                    "multi_modal_data": {"image": img},
                }
            )
        return request_list

    # 设置批大小，可根据显存大小进行调整
    batch_size = 8

    # 用于保存最终结果的列表（长度与 local_dataset 一致）
    dataset_length = len(local_dataset)
    predictions = [None] * dataset_length
    hashed_image_names = [None] * dataset_length

    correct_predictions = 0

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=3
    )

    # 分批处理数据集
    for i in tqdm.trange(0, dataset_length, batch_size):
        batch_data = local_dataset[i : i + batch_size]
        images = batch_data["image"]
        labels = batch_data["label"]

        # 准备批次请求
        requests = prepare_inputs_batch(example_prompt, images)

        # 使用 vLLM 进行推理
        outputs = llm.generate(requests, sampling_params=sampling_params)

        # 对每个输出结果进行处理
        for idx_in_batch, output in enumerate(outputs):
            result = output.outputs[0].text.strip()
            true_label = labels[idx_in_batch]

            global_idx = i + idx_in_batch  # 在数据集中对应的全局索引

            # 计算准确率
            if result == str(true_label):
                correct_predictions += 1

            # 计算图像哈希并保存到本地（可选）
            pil_image = images[idx_in_batch]
            img_bytes = pil_image.tobytes()
            md5_hash = hashlib.md5(img_bytes).hexdigest()
            image_filename = f"{md5_hash}.png"
            image_save_path = os.path.join(images_save_dir, image_filename)
            pil_image.save(image_save_path)

            # 存储推理结果至对应位置
            predictions[global_idx] = result
            hashed_image_names[global_idx] = image_filename

    # 计算准确率
    accuracy = correct_predictions / len(local_dataset)
    print(f"Accuracy: {accuracy:.4f}")

    # 将结果合并回 Dataset
    #   - 新增两列：prediction、hashed_image_name
    #   - 当然也可以再添加其他列，例如置信度等
    local_dataset = local_dataset.add_column("prediction", predictions)
    local_dataset = local_dataset.add_column("hashed_image_name", hashed_image_names)

    # 如果想把准确率等信息也写入 dataset 的 metadata，可以用 info.update
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "accuracy.txt"), "w") as f:
        f.write(f"{accuracy:.4f}")

    # 最终保存整个 Dataset 到指定路径
    local_dataset.save_to_disk(save_path)
    print(f"New dataset saved to: {save_path}")


if __name__ == "__main__":
    main()

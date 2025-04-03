from datasets import load_from_disk
from transformers import AutoTokenizer
from PIL import Image
import torch
from tqdm import tqdm

# 加载数据集和 tokenizer
data = load_from_disk("/aifs4su/yaodong/changye/data/OKVQA_cosi")
tokenizer = AutoTokenizer.from_pretrained("/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf")

def process_sample(sample):
    question = sample["question"]
    choices = sample["choices"]
    activate_list = sample["activate_list"]
    patch_induce_list = sample["patch_induce_list"][0]

    # 构造 prompt
    choice_list_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    prompt = (
        f"<image>\nQuestion: {question}\n"
        f"Options:\n{choice_list_text}\n"
        "Answer with the number of the correct option only. Do not include any other text.\n"
        "Answer:"
    )

    # Tokenize 带 offset
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding["input_ids"][0].tolist()
    offsets = encoding["offset_mapping"][0].tolist()

    # Score 对齐文本部分
    text_start = patch_induce_list[-1] + 1
    score_list = activate_list[text_start:]
    assert len(input_ids) == len(score_list) + 2  # [BOS] + scores + [EOS]

    # ========== 获取 question 区间 ==========
    q_start = prompt.find("Question:") + len("Question:")
    q_end = prompt.find("\nOptions:")

    # ========== 获取 choice 内容区间 ==========
    choice_start = prompt.find("Options:\n") + len("Options:\n")
    choice_end = choice_start + len(choice_list_text)

    # ========== Token Index 匹配 ==========
    question_token_indices = [
        i for i, (start, end) in enumerate(offsets)
        if start >= q_start and end <= q_end
    ]

    choice_token_indices = [
        i for i, (start, end) in enumerate(offsets)
        if start >= choice_start and end <= choice_end
    ]

    # ========== 保护编号和换行符 ==========
    protected_indices = set()

    # 保护 choice 中的 "1.", "2." 等
    for i in range(len(choices)):
        num_prefix = f"{i+1}."
        pos = choice_list_text.find(num_prefix)
        if pos != -1:
            abs_start = choice_start + pos
            abs_end = abs_start + len(num_prefix)
            for j, (start, end) in enumerate(offsets):
                if start >= abs_start and end <= abs_end:
                    protected_indices.add(j)

    # 保护换行符
    for j, (start, end) in enumerate(offsets):
        token_text = prompt[start:end]
        if token_text == "\n":
            protected_indices.add(j)

    # ========== 汇总所有可 mask 的 (token_index, score) ==========
    all_mask_candidates = [
        (i, score_list[i - 2]) for i in question_token_indices + choice_token_indices
        if i not in protected_indices
    ]

    if not all_mask_candidates:
        return {"prompt": prompt}

    # ========== 选后 25% ==========
    all_mask_candidates.sort(key=lambda x: x[1])
    num_to_mask = max(1, int(len(all_mask_candidates) * 0.75))
    tokens_to_mask = set(i for i, _ in all_mask_candidates[:num_to_mask])

    # ========== Mask ==========
    mask_token_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids("<unk>")
    masked_input_ids = input_ids.copy()
    for i in tokens_to_mask:
        masked_input_ids[i] = mask_token_id

    # 解码为字符串
    masked_text = tokenizer.decode(masked_input_ids, skip_special_tokens=False)
    return {"prompt": masked_text}

# ========== 批量处理 ==========
new_data = data.map(process_sample, desc="Masking Q + Choices", num_proc=4)
save_path = "/aifs4su/yaodong/changye/data/OKVQA_cosi_masked_0.25"
new_data.save_to_disk(save_path)
print(f"✅ 全部处理完成，已保存至: {save_path}")

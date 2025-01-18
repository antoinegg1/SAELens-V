export HUGGINGFACE_TOKEN='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'
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli login --token $HUGGINGFACE_TOKEN
# export CURL_CA_BUNDLE=""
# export REQUESTS_CA_BUNDLE=""

# huggingface-cli download htlou/obelics_obelics_100k_tokenized_2048 --local-dir ./obelics_obelics_100k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k_tokenized_2048 --local-dir ./obelics_obelics_10k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_100k --local-dir ./obelics_obelics_100k --repo-type dataset
#!/usr/bin/env bash

# 你要下载的模型仓库列表（只需列出 "htlou/xxx" 的后半部分即可）
repos=(
  "xchen16/CompCap-gpt4"
)

# 你想把文件下载到的主目录
base_dir="/mnt/file2/changye/dataset"

# 确保 huggingface-cli 已经登录
# huggingface-cli login

for repo_name in "${repos[@]}"; do
  echo ">>> Downloading $repo_name ..."
  huggingface-cli download \
    "$repo_name" \
    --local-dir "${base_dir}/${repo_name}" \
    --repo-type dataset \
    # --exclude "slice*"
    # --include "final*" \

  echo
done

echo "All downloads completed!"



# llava-mistral-Align-Anything-L0-q0_25
# llava-mistral-Align-Anything-L0-q0_25_step100
# llava-mistral-RLAIF-V
# llava-mistral-RLAIF-V-Coccur-q0_25_preference
# llava-mistral-RLAIF-V-Coccur-q0_25_preference_step100
# llava-mistral-RLAIF-V-Cosi-q0_25
# llava-mistral-RLAIF-V-Cosi-q0_25_preference
# llava-mistral-RLAIF-V-Cosi-q0_25_preference_step100
# llava-mistral-RLAIF-V-Cosi-q0_50
# llava-mistral-RLAIF-V-Cosi-q0_75
# llava-mistral-RLAIF-V-L0-q0_25_preference
# llava-mistral-RLAIF-V-L0-q0_25_preference_step100
# llava-mistral-RLAIF-V-step200
# llava-mistral-RLAIF-V-step400
# llava-mistral-RLAIF-V-step600
# llava-mistral-RLAIF-V_Coocur-q0_25
# llava-mistral-RLAIF-V_Coocur-q0_25_step100
# llava-mistral-RLAIF-V_Coocur-q0_50
# llava-mistral-RLAIF-V_Coocur-q0_75
# llava-mistral-RLAIF-V_L0-q0_25
# llava-mistral-RLAIF-V_L0-q0_25_step100
# llava-mistral-RLAIF-V_L0-q0_50
# llava-mistral-RLAIF-V_L0-q0_75
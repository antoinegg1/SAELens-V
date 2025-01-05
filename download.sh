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
  # "mm-interp-RLAIF-V-Cosi-q0_25" #p
  # "mm-interp-RLAIF-V-Coccur-q0_25_preference" #4
  # "mm-interp-RLAIF-V_L0-q0_50" #4
  # "mm-interp-RLAIF-V_Coocur-q0_75"  #0
  # "mm-interp-Align-Anything-L0-q0_25"
  # "mm-interp-AA_text_image_to_text"
  # "mm-interp-RLAIF-V-L0-q0_25_preference"
  # "mm-interp-RLAIF-V-Dataset" #0
  # "mm-interp-RLAIF-V-Cosi-q0_75" #run0
  # "mm-interp-RLAIF-V-Cosi-q0_50" #p
  # "mm-interp-RLAIF-V-Cosi-q0_25_preference"
  # "mm-interp-RLAIF-V_L0-q0_75"
  # "mm-interp-RLAIF-V_L0-q0_25"
  # "mm-interp-RLAIF-V_Coocur-q0_50"
  # "mm-interp-RLAIF-V_Coocur-q0_25" #done
  # "mm-interp-Align-Anything-Cosi-q0_25"
  # "mm-interp-Align-Anything-Coccur-q0_25"
  # "mm-interp-AA_preference_l0_0_75"
  # "mm-interp-AA_preference_l0_0_50"
  # "mm-interp-AA_preference_l0_0_25" #done
  # "mm-interp-AA_preference_cosi_0_75"
  # "mm-interp-AA_preference_cosi_0_50"
  # "mm-interp-AA_preference_cosi_0_25"
  # "mm-interp-AA_preference_cocour_q0_25" #done
  # "mm-interp-AA_preference_cocour_0_75" #no file
  # "mm-interp-AA_preference_cocour_0_50"
  # "htlou/saev_anole_obelics_10k"
  # "htlou/saev_chameleon_obelics_100k"
  # "facebook/chameleon-7b"
  # "htlou/mm-interp-AA_preference_cocour_new_step10_0_60"
  # "htlou/mm-interp-AA_preference_cocour_new_step10_0_70"
  # "htlou/mm-interp-AA_preference_cocour_new_step10_0_80"
  # "htlou/mm-interp-AA_preference_cocour_new_step10_0_90"
  # "htlou/mm-interp-AA_preference_cocour_new_step10_0_100"
  # # "htlou/mm-interp-AA_preference_cosi_new_step10_0_80"
  # # "htlou/mm-interp-AA_preference_cosi_new_step10_0_70"
  # # "htlou/mm-interp-AA_preference_cosi_new_step10_0_60"
  # "htlou/mm-interp-AA_preference_cosi_new_step10_0_50"
  # "htlou/mm-interp-AA_preference_cosi_new_step10_0_40"
  # "htlou/mm-interp-AA_preference_cosi_new_step10_0_30"
  # "htlou/mm-interp-AA_preference_cosi_new_step10_0_20"
  # "htlou/mm-interp-AA_preference_cosi_new_step10_0_10"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_80"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_70"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_60"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_50"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_40"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_30"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_20"
  # "htlou/mm-interp-AA_preference_l0_new_step10_0_10"
  # "htlou/mm-interp-AA_preference_random_0_90"
  # "htlou/mm-interp-AA_preference_random_0_80"
  # "htlou/mm-interp-AA_preference_random_0_70"
  # "htlou/mm-interp-AA_preference_random_0_60"
  # "htlou/mm-interp-AA_preference_random_0_50"
  # "htlou/mm-interp-AA_preference_random_0_40"
  # "htlou/mm-interp-AA_preference_random_0_30"
  # "htlou/mm-interp-AA_preference_random_0_20"
  # "htlou/mm-interp-AA_preference_random_0_10"
  "htlou/mm-interp-q0_20_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_10_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_30_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_40_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_50_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_70_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_60_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_80_preference-AA_preference_cocour_new_step10"
  "htlou/mm-interp-q0_90_preference-AA_preference_cocour_new_step10"
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
    --repo-type model \
    --exclude "slice*"
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
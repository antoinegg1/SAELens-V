# export CUDA_VISIBLE_DEVICES=0,1,2,3
step=4000
start=0
total=20000
while [ $start -lt $total ]
do
  python ./cooccurrence.py \
    --model_name llava-hf/llava-v1.6-mistral-7b-hf \
    --model_path "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf" \
    --dataset_path "/aifs4su/yaodong/changye/data/AA_preference" \
    --save_path "/aifs4su/yaodong/changye/data/AA_preference_mistral-7b_interp" \
    --sae_path "/aifs4su/yaodong/changye/model/Antoinegg1/llavasae_obliec100k_SAEV" \
    --batch_size 20 \
    --start_idx $start \
    --end_idx $(($start+$step)) \
    --sae_device "cuda:7"\
    --device "cuda:4" \
    --n_devices 4
  
  start=$(($step+$start))
done

  # python ./cooccurrence.py \
  #   --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  #   --model_path "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-mistral-7b-hf" \
  #   --dataset_path "/aifs4su/yaodong/hantao/datasets/MMInstruct-GPT4V" \
  #   --save_path "/aifs4su/yaodong/changye/data/MMInstruct-GPT4V_mistral-7b_interp" \
  #   --sae_path "/aifs4su/yaodong/changye/model/Antoinegg1/llavasae_obliec100k_SAEV" \
  #   --batch_size 20 \
  #   --start_idx 13000 \
  #   --end_idx 16000 \
  #   --sae_device "cuda:3"\
  #   --device "cuda:0" \
  #   --n_devices 4
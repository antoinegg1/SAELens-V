# export CUDA_VISIBLE_DEVICES=0,1,2,3
step=1000
start=190000
total=201000
while [ $start -lt $total ]
do
  python ./cooccurrence.py \
    --model_name llava-hf/llava-v1.6-vicuna-7b-hf \
    --model_path "/aifs4su/yaodong/changye/model/llava-hf/llava-v1.6-vicuna-7b-hf" \
    --dataset_path "/aifs4su/yaodong/changye/data/AA_preference" \
    --save_path "/aifs4su/yaodong/changye/data/AA_preference_vicuna-7b_interp" \
    --sae_path "/aifs4su/yaodong/changye/checkpoints-V7/xnsl8657/final_122880000" \
    --batch_size 20 \
    --start_idx $start \
    --end_idx $(($start+$step)) \
     --sae_device "cuda:7"\
    --device "cuda:4" \
    --n_devices 4
  
  start=$(($start+$step))
done
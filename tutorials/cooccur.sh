# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7
step=4000
total=120000
while [ $step -lt $total ]
do
  python ./cooccurrence.py \
    --model_path "/mnt/file2/changye/model/llava" \
    --dataset_path "/mnt/file2/changye/dataset/CompCap-gpt4/data" \
    --save_path "/mnt/file2/changye/dataset/Compcap_interp" \
    --sae_path "/mnt/file2/changye/model/llavasae_obliec100k_SAEV" \
    --batch_size 10 \
    --start_idx $step \
    --end_idx $(($step+4000)) 
  
  step=$(($step+4000))
done


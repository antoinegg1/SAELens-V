Dataset_path="/mnt/file2/changye/dataset/Align-Anything_preference_1k"
Output_base_dir="/mnt/file2/changye/dataset/interp/Align-Anything-preference_chameleon_interp"
Model_base_dir="/mnt/file2/changye/model/htlou/chameleon"
SAE_path="/mnt/file2/changye/model/SAE/saev_chameleon_obelics_100k/final_122880000"
Model_name="htlou/AA-Chameleon-7B-plus"
Feature_num=131072

for model in $Model_base_dir/*; do
  echo $model
  output_dir=$Output_base_dir/$(basename $model)
  echo $output_dir
  python /mnt/file2/changye/SAELens-V/tutorials/cosimilarity_chameleon.py \
    --model_name $Model_name \
    --model_path $model \
    --sae_path $SAE_path \
    --dataset_path $Dataset_path \
    --output_dir $output_dir \
    --feature_num $Feature_num
done 

# python /mnt/file2/changye/SAELens-V/tutorials/cosimilarity_chameleon.py \
#   --model_name "htlou/AA-Chameleon-7B-plus" \
#   --model_path "/mnt/file2/changye/model/facebook/chameleon-7b" \
#   --sae_path "/mnt/file2/changye/model/SAE/saev_chameleon_obelics_100k/final_122880000" \
#   --dataset_path "/mnt/file2/changye/dataset/Align-Anything_preference_1k" \
#   --output_dir "/mnt/file2/changye/dataset/interp/Align-Anything-preference_chameleon_interp/chameleon-7b_preference" \
#   --feature_num 131072 \


  
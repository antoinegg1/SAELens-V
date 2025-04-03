
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
# import pdb;pdb.set_trace()
import os
# os.environ["HF_HOME"] = "/tmp/hf"
# os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf/transformers"
# os.environ["HF_DATASETS_CACHE"] = "/tmp/hf/datasets"
# os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets"

cfg = PretokenizeRunnerConfig(
    tokenizer_name="lmsys/vicuna-7b-v1.5",
    dataset_path="/aifs4su/yaodong/changye/data/NeelNanda/pile-10k", # this is just a tiny test dataset
    # data_files={"train": "/aifs4su/yaodong/changye/data/obelics_100k/obelics_100k_washed.json"},
    shuffle=True,
    num_proc=64, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=4096,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    # image_column_name="images",
    column_name="text",
    # uncomment to upload to huggingface
    # hf_repo_id="your-username/c4-10k-tokenized-gpt2"

    # uncomment to save the dataset locally
    save_path="/aifs4su/yaodong/changye/data/pile10k-tokenized_vicuna-7B_4096"
)

dataset = PretokenizeRunner(cfg).run()
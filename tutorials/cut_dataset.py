from datasets import load_from_disk, concatenate_datasets, Dataset
import argparse
import os
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
def split_dataset_into_shards(sorted_dataset, num_splits):
    shards = []
    for i in range(num_splits):
        shard = sorted_dataset.shard(num_shards=num_splits, index=i, contiguous=True)
        shards.append(shard)
    return shards
def setup_logging():
    """
    配置日志记录格式和级别。
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def save_split(split_info):
    """
    保存数据集分割部分的函数，用于并行执行。
    
    Args:
        split_info (tuple): 包含分割数据集和保存路径的元组。
        
    Returns:
        str: 保存成功的路径或错误信息。
    """
    split_dataset, save_path = split_info
    try:
        split_dataset.save_to_disk(save_path)
        return f"成功保存到 {save_path}"
    except Exception as e:
        return f"保存到 {save_path} 时出错: {e}"

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="合并、排序并拆分大型数据集。"
    )
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="包含多个数据集的输入文件夹路径。"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True, 
        help="保存拆分后数据集的输出文件夹路径。"
    )
    parser.add_argument(
        "--sort_column", 
        type=str, 
        default="Cooccur_score", 
        help="用于排序的数据列名称。默认值为'Cooccur_score'。"
    )
    parser.add_argument(
        "--num_splits", 
        type=int, 
        default=10, 
        help="将数据集拆分成的部分数量。默认值为10。"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="并行保存时使用的最大工作进程数。默认使用CPU核心数。"
    )
    args = parser.parse_args()

    # 设置最大工作进程数
    if args.max_workers is None:
        max_workers = multiprocessing.cpu_count()
    else:
        max_workers = args.max_workers

    os.makedirs(args.output_folder, exist_ok=True)

    # 加载数据集
    logging.info("开始加载数据集...")
    dataset_list = []
    input_folders = [
        os.path.join(args.input_folder, fold) 
        for fold in os.listdir(args.input_folder) 
        if os.path.isdir(os.path.join(args.input_folder, fold))
    ]

    if not input_folders:
        logging.error(f"在输入文件夹中未找到任何数据集: {args.input_folder}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(load_from_disk, folder): folder for folder in input_folders}
        for future in tqdm.tqdm(as_completed(future_to_folder), total=len(future_to_folder), desc="加载数据集"):
            folder = future_to_folder[future]
            try:
                dataset = future.result()
                dataset_list.append(dataset)
                logging.info(f"成功加载数据集: {folder}")
            except Exception as e:
                logging.error(f"加载数据集 {folder} 失败: {e}")

    if not dataset_list:
        logging.error("未加载到任何数据集。请检查输入文件夹路径和内容。")
        return

    logging.info(f"共加载了 {len(dataset_list)} 个数据集。")

    # 合并数据集
    logging.info("开始合并数据集...")
    try:
        dataset = concatenate_datasets(dataset_list)
        logging.info(f"数据集合并完成，总样本数: {len(dataset)}")
    except Exception as e:
        logging.error(f"合并数据集时出错: {e}")
        return

    # 排序数据集
    logging.info(f"按 '{args.sort_column}' 列进行降序排序...")
    try:
        sorted_dataset = dataset.sort(args.sort_column, reverse=True)
        logging.info("数据集排序完成。")
    except Exception as e:
        logging.error(f"排序数据集时出错: {e}")
        return

    # 使用 Dataset.shard 进行分割
    logging.info("开始使用 Dataset.shard 方法进行数据集分割...")
    try:
        shards = split_dataset_into_shards(sorted_dataset, args.num_splits)
        logging.info(f"数据集已分割为 {args.num_splits} 个部分。")
    except Exception as e:
        logging.error(f"分割数据集时出错: {e}")
        return
    # 准备保存信息
    split_infos = []
    for i, shard in enumerate(shards):
        save_path = os.path.join(args.output_folder, f"top_{(i+1)*10}%.dataset")
        split_infos.append((shard, save_path))
    # 并行保存分割数据集
    logging.info(f"将数据集拆分成 {args.num_splits} 个部分并保存...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(save_split, info): info for info in split_infos}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="保存分割部分"):
            result = future.result()
            if result.startswith("成功"):
                logging.info(result)
            else:
                logging.error(result)

    logging.info("所有拆分部分已成功保存。")

if __name__ == "__main__":
    main()

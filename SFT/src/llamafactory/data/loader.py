# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import glob
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk, concatenate_datasets

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    # 如果指定了 tokenized_path，启用增量保存模式
    incremental_save = (
        data_args.tokenized_path is not None 
        and not data_args.streaming 
        and training_args.should_save
        and isinstance(dataset, Dataset)  # 只对 Dataset 类型启用，不支持 IterableDataset
        and hasattr(dataset, "__len__")
    )
    
    if incremental_save:
        # 增量保存模式：将数据集分成多个 shard，每个 shard 处理完后立即保存
        dataset_len = len(dataset)
        # 每个 shard 的大小：默认每 10000 条保存一次，可根据数据集大小调整
        shard_size = min(dataset_len//10 + 1, 1000)
        num_shards = (dataset_len + shard_size - 1) // shard_size
        
        # 创建临时目录用于保存 shard
        # Multiple paths are not allowed when saving
        base_path = data_args.tokenized_path
        if isinstance(base_path, list):
            if len(base_path) > 1:
                raise ValueError(
                    "Multiple paths are not allowed when saving tokenized datasets. "
                    "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
                )
            base_path = base_path[0] if len(base_path) > 0 else None
        elif isinstance(base_path, str) and "," in base_path:
            raise ValueError(
                "Multiple paths (comma-separated) are not allowed when saving tokenized datasets. "
                "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
            )
        
        if base_path is None:
            incremental_save = False
        else:
            shard_dir = f"{base_path}_shards"
            os.makedirs(shard_dir, exist_ok=True)
            
            # 检查是否所有 shard 都已存在
            all_shards_exist = all(
                os.path.exists(os.path.join(shard_dir, f"shard_{i:04d}"))
                for i in range(num_shards)
            )
            
            if all_shards_exist and not data_args.overwrite_cache:
                # 所有 shard 都已存在，直接加载并合并
                logger.info_rank0(f"检测到所有 {num_shards} 个 shard 已处理完成，直接加载...")
                processed_shards = []
                for shard_idx in range(num_shards):
                    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}")
                    try:
                        shard_dataset = load_from_disk(shard_path)
                        processed_shards.append(shard_dataset)
                    except Exception as e:
                        logger.warning_rank0(f"加载 shard {shard_idx} 失败，将重新处理: {e}")
                        all_shards_exist = False
                        break
                
                if all_shards_exist and len(processed_shards) == num_shards:
                    logger.info_rank0(f"成功加载所有 {num_shards} 个 shard，正在合并...")
                    dataset = concatenate_datasets(processed_shards)
                    logger.info_rank0("所有 shard 合并完成")
                    return dataset
            
            # 需要处理 shard（部分或全部）
            processed_shards = []
            for shard_idx in range(num_shards):
                start_idx = shard_idx * shard_size
                end_idx = min((shard_idx + 1) * shard_size, dataset_len)
                shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}")
                
                # 检查 shard 是否已经处理过
                if os.path.exists(shard_path) and not data_args.overwrite_cache:
                    logger.info_rank0(f"加载已处理的 shard {shard_idx + 1}/{num_shards} (索引 {start_idx}-{end_idx})...")
                    try:
                        shard_dataset = load_from_disk(shard_path)
                        processed_shards.append(shard_dataset)
                        continue
                    except Exception as e:
                        logger.warning_rank0(f"加载 shard {shard_idx} 失败，将重新处理: {e}")
                
                # 处理当前 shard
                logger.info_rank0(f"处理 shard {shard_idx + 1}/{num_shards} (索引 {start_idx}-{end_idx})...")
                shard = dataset.select(range(start_idx, end_idx))
                
                shard = shard.map(
                    dataset_processor.preprocess_dataset,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=column_names,
                    **kwargs,
                )
                
                # 检查 shard 是否为空
                if len(shard) == 0:
                    logger.warning_rank0(f"Shard {shard_idx + 1}/{num_shards} (索引 {start_idx}-{end_idx}) 处理后的数据为空，跳过保存。这可能是因为数据预处理时所有数据都被过滤掉了。")
                    continue
                
                # 立即保存当前 shard
                shard.save_to_disk(shard_path)
                logger.info_rank0(f"Shard {shard_idx + 1}/{num_shards} 已保存到 {shard_path}")
                processed_shards.append(shard)
            
            # 合并所有 shard
            logger.info_rank0(f"合并 {len(processed_shards)} 个 shard...")
            dataset = concatenate_datasets(processed_shards)
            logger.info_rank0("所有 shard 合并完成")
    else:
        # 原有的处理方式
        dataset = dataset.map(
            dataset_processor.preprocess_dataset,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=column_names,
            **kwargs,
        )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        # Helper function to check if multiple paths are provided
        def is_multiple_paths(path):
            if isinstance(path, list):
                return len(path) > 1
            elif isinstance(path, str):
                return "," in path
            return False
        
        # Helper function to normalize paths to list
        def normalize_paths(path):
            if isinstance(path, str):
                # Check if it's a comma-separated string
                if "," in path:
                    return [p.strip() for p in path.split(",")]
                else:
                    return [path]
            elif isinstance(path, list):
                return path
            else:
                return [path]
        
        # Support multiple paths: list, comma-separated string, or single path
        tokenized_paths = normalize_paths(data_args.tokenized_path)
        
        # Check if all paths have tokenized data
        valid_paths = [p for p in tokenized_paths if has_tokenized_data(p)]

        # logger.info_rank0(f"valid_paths: {valid_paths}")  # debug if needed
        
        if len(valid_paths) == 0:
            if data_args.streaming:
                raise ValueError("Turn off `streaming` when saving dataset to disk.")
        else:
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            
            # Load all datasets
            all_datasets = []
            all_eval_datasets = []
            
            for path in valid_paths:
                tokenized_data = load_from_disk(path)
                dataset_module = get_dataset_module(tokenized_data)
                
                if dataset_module.get("train_dataset") is not None:
                    all_datasets.append(dataset_module["train_dataset"])
                
                if dataset_module.get("eval_dataset") is not None:
                    eval_ds = dataset_module["eval_dataset"]
                    if isinstance(eval_ds, dict):
                        all_eval_datasets.append(eval_ds)
                    else:
                        all_eval_datasets.append(eval_ds)
            
            # Combine train datasets
            if len(all_datasets) > 0:
                if len(all_datasets) == 1:
                    combined_train = all_datasets[0]
                else:
                    # Use merge_dataset to combine multiple datasets
                    combined_train = merge_dataset(all_datasets, data_args, training_args.seed)
            else:
                combined_train = None
            
            # Combine eval datasets
            combined_eval = None
            if len(all_eval_datasets) > 0:
                if len(all_eval_datasets) == 1:
                    combined_eval = all_eval_datasets[0]
                else:
                    # For eval datasets, if they are all dicts, merge them
                    # Otherwise, concatenate non-dict datasets and merge dicts separately
                    eval_dicts = [ds for ds in all_eval_datasets if isinstance(ds, dict)]
                    eval_datasets = [ds for ds in all_eval_datasets if not isinstance(ds, dict)]
                    
                    if len(eval_dicts) > 0 and len(eval_datasets) > 0:
                        # Mix of dicts and datasets: merge dicts and concatenate datasets
                        combined_eval = {}
                        for eval_dict in eval_dicts:
                            combined_eval.update(eval_dict)
                        # Add concatenated datasets as a single entry
                        if len(eval_datasets) > 1:
                            combined_eval["combined"] = concatenate_datasets(eval_datasets)
                        else:
                            combined_eval["combined"] = eval_datasets[0]
                    elif len(eval_dicts) > 0:
                        # All are dicts: merge them
                        combined_eval = {}
                        for eval_dict in eval_dicts:
                            combined_eval.update(eval_dict)
                    else:
                        # All are datasets: concatenate them
                        combined_eval = concatenate_datasets(eval_datasets)
            
            # Create final dataset module
            dataset_module: "DatasetModule" = {}
            if combined_train is not None:
                dataset_module["train_dataset"] = combined_train
            if combined_eval is not None:
                dataset_module["eval_dataset"] = combined_eval
            
            if data_args.streaming and dataset_module.get("train_dataset") is not None:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()
            
            logger.info_rank0(f"Loaded tokenized datasets from {len(valid_paths)} path(s): {valid_paths}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                # Multiple paths are not allowed when saving
                save_path = data_args.tokenized_path
                if isinstance(data_args.tokenized_path, list):
                    if len(data_args.tokenized_path) > 1:
                        raise ValueError(
                            "Multiple paths are not allowed when saving tokenized datasets. "
                            "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
                        )
                    save_path = data_args.tokenized_path[0] if len(data_args.tokenized_path) > 0 else None
                elif isinstance(data_args.tokenized_path, str) and "," in data_args.tokenized_path:
                    raise ValueError(
                        "Multiple paths (comma-separated) are not allowed when saving tokenized datasets. "
                        "Please provide only a single path for saving, or use multiple paths only when loading existing datasets."
                    )
                
                if save_path is not None:
                    dataset_dict.save_to_disk(save_path)
                    print(f"Tokenized dataset is saved at {save_path}.")
                    logger.info_rank0(f"Tokenized dataset is saved at {save_path}.")
                    logger.info_rank0(f"Please launch the training with `tokenized_path: {save_path}`.")
                
                # 清理之前保存的 shard 目录
                base_path = save_path if save_path is not None else data_args.tokenized_path
                shard_dir = f"{base_path}_shards"
                
                if os.path.exists(shard_dir):
                    try:
                        shutil.rmtree(shard_dir)
                        logger.info_rank0(f"已清理 shard 目录: {shard_dir}")
                    except Exception as e:
                        logger.warning_rank0(f"清理 shard 目录 {shard_dir} 失败: {e}")
                
                # 清理转换后的数据集目录（格式: {base_path}_converted_{hash}）
                # 转换后的数据集路径格式为: {base_path}_converted_{hash}
                converted_pattern = f"{base_path}_converted_*"
                converted_dirs = glob.glob(converted_pattern)
                
                for converted_dir in converted_dirs:
                    if os.path.isdir(converted_dir):
                        try:
                            shutil.rmtree(converted_dir)
                            logger.info_rank0(f"已清理转换后的数据集目录: {converted_dir}")
                        except Exception as e:
                            logger.warning_rank0(f"清理转换后的数据集目录 {converted_dir} 失败: {e}")

        return get_dataset_module(dataset_dict)

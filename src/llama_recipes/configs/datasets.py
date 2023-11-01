# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class tabledetection_dataset:
    dataset: str = 'tabledetection_dataset'
    train_split: str = 'train'
    test_split: str = 'teset_263'
    data_path: str = 'src/llama_recipes/datasets/tabledetection_dataset'


@dataclass
class tablesense_dataset_1:
    dataset: str = 'tablesense_dataset_1'
    train_split: str = "train"
    test_split: str = "test_263"
    data_path: str = "src/llama_recipes/datasets/tablesense_dataset/subtask_all"


@dataclass
class tablesense_dataset_2:
    dataset: str = 'tablesense_dataset_2'
    train_split: str = "train"
    test_split: str = "test_263"
    data_path: str = "src/llama_recipes/datasets/tablesense_dataset/subtask_all"


@dataclass
class tablesense_dataset_3:
    dataset: str = 'tablesense_dataset_3'
    train_split: str = "train"
    test_split: str = "test_263"
    data_path: str = "src/llama_recipes/datasets/tablesense_dataset/subtask_all"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
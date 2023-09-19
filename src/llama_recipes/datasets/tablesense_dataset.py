import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List


class TableSenseDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2000):
        self.tables = json.load(open(os.path.join(dataset_config.data_path, partition + "_row_feature.json")))
        self.tokenizer = tokenizer
        self.max_words = max_words

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        table_item = self.tables[index]
        prompt = table_item['prompt']
        example = prompt + table_item['answer']
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        assert padding > 0
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
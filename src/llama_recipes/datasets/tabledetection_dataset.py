import copy
import json
import os
import torch
import jsonlines

from torch.utils.data import Dataset
from pathlib import Path


class TableDetectionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=4000, subtask_index=1):
        self.tables = []
        with open(os.path.join(dataset_config.data_path, partition + ".jsonl")) as json_file:
            for line in json_file.readlines():
                self.tables.append(json.loads(line))
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
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.to_list(),
            "labels": labels.to_list(),
            "attention_mask": example_mask.to_list(),
        }
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import json
import os
import re
import sys
import time
import random
from typing import List
from pathlib import Path
from tqdm import tqdm
from transformers import LlamaTokenizer, default_data_collator
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model, load_llama_from_config
from llama_recipes.configs.datasets import tablesense_dataset
from llama_recipes.configs import train_config
from llama_recipes.utils.config_utils import generate_dataset_config
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset


def main(
        model_name,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 0.1,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
        enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
        enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
        max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False,
        is_multi: bool = False,
        is_dev: bool = False,
        # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):
    user_prompt_list = json.load(open(os.path.join(tablesense_dataset.data_path,
                                                   "train_row_feature.json" if is_dev else "test_263_row_feature.json"), 'r'))
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    with open(os.path.join(Path(__file__).parent, 'settings.json'), 'r') as f:
        settings = json.load(f)
        special_tokens = settings['special_tokens']
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model.resize_token_embeddings(model.config.vocab_size + 1 + len(special_tokens))

    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if is_multi:
        error_log = open('inference_error.txt', 'w')
    random.shuffle(user_prompt_list)
    for data in valid_dataloader:
        user_prompt = data['prompt']
        batch = tokenizer(user_prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not is_multi:
            print(f"Input: \n {user_prompt}")
            print(f"Answer: {data['answer']}")
            print(f"Model Output: \n {output_text}")
            return

        wrong_format_count = 0
        wrong_answer_count = 0
        if not output_text.find('Number: '):
            wrong_format_count += 1
            error_log.write(output_text)
        else:
            pattern = r'\d+'
            match = re.findall(pattern, output_text)
            reality = re.findall(pattern, data['answer'])
            if int(match[-1]) == int(reality[-1]):
                wrong_answer_count += 1
            else:
                error_log.write(output_text)

    error_log.close()
    print(f"Wrong Answer: {wrong_answer_count}")
    print(f"Wrong Format: {wrong_format_count}")


if __name__ == "__main__":
    fire.Fire(main)

import json
import os
from llama_recipes.configs.datasets import tablesense_dataset

def create_test_file():
    user_prompt_list = json.load(open(os.path.join(tablesense_dataset.data_path, "subtask_all", "test_263_row_feature.json"),
                                      'r'))



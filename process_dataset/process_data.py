import jsonlines
import os
import json
import fire
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)


# read SETTINGS
with open(os.path.join(Path(__file__).parent.parent, 'settings.json'), 'r') as settings_file:
    settings_json = json.load(settings_file)
    limit = settings_json['limit']
    template1 = settings_json['prompt_template1']
    template2 = settings_json['prompt_template2']
    template3 = settings_json['prompt_template3']
    special_tokens = settings_json['special_tokens']
save_prefix = "/spot/v-qinyuxu/"
save_folder = "llama_dataset/"


def get_type(header_num, middle_homo, final_AGG):
    if header_num == 1 and middle_homo is False:
        if final_AGG is True:
            return 2
        else:
            return 1
    else:
        return 3


# read table from dataset
def get_full_table():
    with open(save_prefix + 'SavedFileNew/' + 'test_full.txt', 'r') as f:
        tables = f.readlines()
    with open(save_prefix + 'SavedFileNew/' + 'train_full.txt', 'r') as f:
        tables.extend(f.readlines())
    dic = {}
    index = 0
    next_index = 1
    while index < len(tables):
        while next_index < len(tables) and '<begin>' not in tables[next_index]:
            next_index += 1
        specifier, tag = tables[index].split('<begin>')
        for tmp_index in range(index + 1, next_index):
            tag += tables[tmp_index]
        assert specifier.startswith('\\\\')
        dic[specifier] = tag.replace('\n', '')
        index = next_index
        next_index += 1
    return dic


# change table to answer form
def to_markdown_table(rows):
    res = ""
    for row in rows:
        res += "<tr>" + row + "</tr>"
    return res


# get answer from tags and rows
def get_answer(tags, rows, index):
    num_header = 0
    for tag in tags:
        if tag == "SND" or tag == "HEADER":
            num_header += 1
        else:
            break
    if index == 1:
        return num_header
    elif index == 2:
        is_have = False
        if (tags[num_header:].count("BOD") > 1 or tags[num_header:-1].count("AGG") > 0
                or tags[num_header:].count("SEC") > 0 or tags[num_header:].count("SND") > 0):
            is_have = True
        return is_have
    elif index == 3:
        return tags[-1] == "AGG"


def transform_json(data):
    return {"type": get_type(data['answer1'], data['answer2'], data['answer3']),
            'prompt1': template1.format(data['rows']),
            'answer1': "Answer: " + str(data['answer1']),
            'prompt2': template2.format(data['rows']),
            'answer2': "Answer: " + str(data['answer2']),
            'prompt3': template3.format(data['rows']),
            'answer3': "Answer: " + str(data['answer3']),
            }


def process_long_snd(rows, tags, tokenizer):
    to_include = []
    is_snd = True
    is_bod = False
    is_agg = False
    length = (150  # length of the template
              + torch.tensor(tokenizer.encode(to_markdown_table([rows[0]])), dtype=torch.int64).shape[0]
              + 20)
    for index, tag in enumerate(tags):
        if is_snd and not is_bod and tag == "BOD" and length + torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1 <= limit:
            to_include.append(index)
            length += torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1
        elif is_snd and is_bod and length + torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1 <= limit:
            to_include.append(index)
            length += torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1
        elif is_snd and is_bod and is_agg and tag =="AGG" and length + torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1 <= limit:
            to_include.append(index)
            length += torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1

    to_include = list(set(to_include))
    to_include.sort()
    rows = [rows[i] for i in to_include]
    tags = [tags[i] for i in to_include]

    if len(to_include) <= 2:
        print("NOOO")
        return None
    if length > limit:
        return None

    assert len(tags) == len(rows)
    # assert get_answer(tags, rows, subtask_index) == get_answer(ori_tags, rows, subtask_index)
    return {
        "rows": to_markdown_table(rows),
        "answer1": get_answer(tags, rows, 1),
        "answer2": get_answer(tags, rows, 2),
        "answer3": get_answer(tags, rows, 3),
    }



def extract_from_table(rows, tags, tokenizer):
    # for all the rows starting from the header down to the second to last row
    # BOD: it's necessary to extract at least one BOD and one DAT in this data section; SND: include it
    # BLA: discard it; AGG: keep it

    ori_tags = tags
    num_header = 0
    for tag in tags:
        if tag == 'SND':
            num_header += 1
        else:
            break

    to_include = list(range(num_header))
    length = (150  # length of the template
              +torch.tensor(tokenizer.encode(to_markdown_table(rows[:num_header])), dtype=torch.int64).shape[0]
              +20)  # length of answer

    if tags[-1] != "SND":
        to_include.append(len(tags) - 1)
        length += torch.tensor(tokenizer.encode(to_markdown_table([rows[-1]])), dtype=torch.int64).shape[0]

    if length > limit:
        return process_long_snd(rows, tags, tokenizer)
    is_non_snd = False
    for index, tag in enumerate(tags[num_header:-1], start=num_header):
        if tag == "SND" and is_non_snd is False:
            continue
        if tag == "AGG" or tag == "SND" or tag == "SEC":
            new_len = torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1
            if length + new_len <= limit:
                is_non_snd = True
                to_include.append(index)
                length += new_len
        if tag == "BOD":
            new_len = torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1
            if length + new_len <= limit:
                is_non_snd = True
                to_include.append(index)
                length += new_len
    index = num_header
    while index < len(tags) - 1:
        if to_include.count(index) != 0:
            index += 1
        elif tags[index] == "SND" and is_non_snd is False:
            index += 1
        elif length + torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1 <= limit:
            to_include.append(index)
            length += torch.tensor(tokenizer.encode(to_markdown_table([rows[index]])), dtype=torch.int64).shape[0] + 1
            index += 1
        else:
            break

    to_include = list(set(to_include))
    to_include.sort()
    rows = [rows[i] for i in to_include]
    tags = [tags[i] for i in to_include]

    if len(to_include) <= 2:
        return None

    assert len(tags) == len(rows)
    # assert get_answer(tags, rows, subtask_index) == get_answer(ori_tags, rows, subtask_index)
    return {
        "rows": to_markdown_table(rows),
        "answer1": get_answer(tags, rows, 1),
        "answer2": get_answer(tags, rows, 2),
        "answer3": get_answer(tags, rows, 3),
    }


def get_output_from_table_one(original_feature_one, dic_specifier_to_row, tokenizer):
    # process original feature file and add specifier to the output file
    tmp_split_feature = original_feature_one.replace('\n', '').split('|')

    tags = []
    for line_original_features in tmp_split_feature[6:]:
        tags.append(line_original_features.split(';')[-1])

    # try to get the original table according to the specifier
    try:
        list_of_row_original_table = dic_specifier_to_row['|'.join(tmp_split_feature[:6])].split('<newline>')
        list_of_row_original_table = list(filter(lambda x: str(x).strip() != '', list_of_row_original_table))
        list_of_row_original_table = list(map(lambda x: x.replace('\n', ''), list_of_row_original_table))
    except Exception as _:
        print("ERROR!")
        return []

    assert all(['<begin>' not in row for row in list_of_row_original_table])
    assert len(list_of_row_original_table) == len(tags)
    output = to_markdown_table(list_of_row_original_table)
    length = torch.tensor(tokenizer.encode(output), dtype=torch.int64).shape[0]
    if length + 150 + 20 <= limit:
        return {
            "rows": output,
            "answer1": get_answer(tags, list_of_row_original_table, 1),
            "answer2": get_answer(tags, list_of_row_original_table, 2),
            "answer3": get_answer(tags, list_of_row_original_table, 3)
        }
    else:
        return extract_from_table(list_of_row_original_table, tags, tokenizer)


def run(feature_file, tokenizer, save_dir):
    # get input
    err_count = 0
    dic = get_full_table()
    sample = open(save_prefix + save_folder + save_dir + '/' + 'sample.txt', 'w')
    with open(save_prefix + 'original_feature/' + feature_file + '.txt', 'r') as f:
        tables = f.readlines()

    # get output
    output = []
    for index, table in tqdm(enumerate(tables)):
        data = get_output_from_table_one(table, dic, tokenizer)
        if data:
            sample.write('\n'.join(data['rows']) + '\n' + str(data['answer1']) + '  ' + str(data['answer2']) + '  ' + str(data['answer3']) + '  ' + '\n\n\n')
            data_json = transform_json(data)
            length = torch.tensor(tokenizer.encode(data['rows']), dtype=torch.int64).shape[0]
            if length + 20 + 150 > limit:
                err_count += 1
                print(index)
                continue
            output.append(data_json)
        else:
            print(index)
            err_count += 1
    print("Error: " + str(err_count))

    # write into one json file
    sample.close()
    # random.shuffle(output)
    f = jsonlines.open(save_prefix + save_folder + save_dir + '/' + feature_file + '.json', 'w')
    jsonlines.Writer.write(f, output)


def main(
        model_name: str,
        save_dir: str,
):
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join('/spot/v-qinyuxu', model_name))
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    for file in ['test_263_row_feature', 'train_row_feature', 'valid_row_feature']:
        run(file, tokenizer, save_dir)


if __name__ == '__main__':
    fire.Fire(main)

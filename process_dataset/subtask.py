import jsonlines
import os
import json
import fire
from tqdm import tqdm
from pathlib import Path


# read SETTINGS
with open(os.path.join(Path(__file__).parent.parent, 'settings.json'), 'r') as settings_file:
    settings_json = json.load(settings_file)
    limit = settings_json['limit']
    template1 = settings_json['prompt_template1']
    template2 = settings_json['prompt_template2']
    template3 = settings_json['prompt_template3']
save_prefix = "/spot/v-qinyuxu/"
save_folder = "llama_dataset/"


def get_type(header_num, middle_homo, final_AGG):
    if header_num == 1 and middle_homo is True:
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
        if (tags[num_header:-1].count("BOD") > 1 or tags[num_header:-1].count("AGG") > 0
                or tags[num_header:-1].count("SEC") > 0):
            is_have = True
        return is_have
    elif index == 3:
        return tags[-1] == "AGG"


def transform_json(data):
    return {'prompt1': template1.format(data['rows']),
            'answer1': "Answer: " + str(data['answer1']),
            'prompt2': template2.format(data['rows']),
            'answer2': "Answer: " + str(data['answer2']),
            'prompt3': template3.format(data['rows']),
            'answer3': "Answer: " + str(data['answer3']),
            "type": get_type(data['answer1'], data['answer2'], data['answer3'])
            }


def extract_from_table(rows, tags,):
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
    length = (  150  # length of the template
              + len(to_markdown_table(rows[:num_header]))  # length of the table
              + 10)  # length of answer

    if tags[-1] != "SND":
        to_include.append(len(tags) - 1)
        length += len(to_markdown_table([rows[-1]]))

    if length > limit:
        return None
    is_non_snd = False
    for index, tag in enumerate(tags[num_header:-1], start=num_header):
        if tag == "SND" and is_non_snd is False:
            continue
        if tag == "AGG" or tag == "SND" or tag == "SEC":
            if length + len(to_markdown_table([rows[index]])) <= limit:
                is_non_snd = True
                to_include.append(index)
                length += len(to_markdown_table([rows[index]]))
        if tag == "BOD":
            if length + len(to_markdown_table([rows[index]])) <= limit:
                is_non_snd = True
                to_include.append(index)
                length += len(to_markdown_table([rows[index]])) + 1
    index = num_header
    while index < len(tags) - 1:
        if to_include.count(index) != 0:
            index += 1
        elif tags[index] == "SND" and is_non_snd is False:
            index += 1
        elif length + len(to_markdown_table([rows[index]])) <= limit:
            to_include.append(index)
            length += len(to_markdown_table([rows[index]]))
            index += 1
        else:
            break

    to_include = list(set(to_include))
    to_include.sort()
    rows = [rows[i] for i in to_include]
    tags = [tags[i] for i in to_include]

    if len(to_include) <= 2:
        return None

    # assert get_answer(tags, rows, subtask_index) == get_answer(ori_tags, rows, subtask_index)
    return {
        "rows": rows,
        "answer1": get_answer(tags, rows, 1),
        "answer2": get_answer(tags, rows, 2),
        "answer3": get_answer(tags, rows, 3),
    }


def get_output_from_table_one(original_feature_one, dic_specifier_to_row):
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
    if 150 + len(output) + 20 <= limit:
        return {
            "rows": list_of_row_original_table,
            "answer1": get_answer(tags, list_of_row_original_table, 1),
            "answer2": get_answer(tags, list_of_row_original_table, 2),
            "answer3": get_answer(tags, list_of_row_original_table, 3)
        }
    else:
        return extract_from_table(list_of_row_original_table, tags)


def run(feature_file):
    # get input
    err_count = 0
    dic = get_full_table()
    sample = open(save_prefix + save_folder + 'sample.txt', 'w')
    with open(save_prefix + 'original_feature/' + feature_file + '.txt', 'r') as f:
        tables = f.readlines()


    # get output
    output = []
    for table in tqdm(tables):
        data = get_output_from_table_one(table, dic)
        if data:
            sample.write(str(data['rows']) + '\n' + data['answer1'] + '  ' + data['answer2'] + '  ' + data['answer3'] + '  ')
            data_json = transform_json(data)
            if len(data['rows']) + 20 + 150 > limit:
                err_count += 1
                continue
            output.append(data_json)
            # if data['answer'] > 2:
            #     data_da = transform_json(delete_snd(data))
            #     output.append(data_da)
            #     sample.write(data_json['prompt'] + '\n' + data_json['answer'] + '\n')
            #     sample.write(data_da['prompt'] + '\n' + data_da['answer'] + '\n\n')
        else:
            err_count += 1
    print(err_count)

    # write into one json file
    sample.close()
    # random.shuffle(output)
    f = jsonlines.open(save_prefix + save_folder + feature_file + '.json', 'w')
    jsonlines.Writer.write(f, output)


def main():
    for file in ['train_row_feature', 'test_263_row_feature']:
        run(file)


if __name__ == '__main__':
    fire.Fire(main)

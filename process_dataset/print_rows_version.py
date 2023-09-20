from tqdm import tqdm
import jsonlines
from prompt_template import prompt_template

# SETTINGS
save_prefix = "/spot/v-qinyuxu/"

limit = 3100


def to_markdown_table(rows):
    return '\n'.join(rows)


def get_answer(tags, rows):
    num_header = 0
    for tag in tags:
        if tag == "SND" or tag == "HEADER":
            num_header += 1
        else:
            break

    return 'Number:' + str(num_header) + ';Rows: ' + '\n'.join(rows[:num_header])


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
        for tmp_index in range(index+1, next_index):
            tag += tables[tmp_index]
        assert specifier.startswith('\\\\')
        dic[specifier] = tag.replace('\n', '')
        index = next_index
        next_index += 1
    return dic


def extract_from_table(rows, tags):
    # for all the rows starting from the header down to the second to last row
    # if BOD exists, then it's necessary to extract at least one BOD and one DAT in this data section
    # if SND exists, then include it
    # if BLA exists, then discard it
    # if AGG exists, then keep it

    ori_tags = tags
    num_header = 0
    for tag in tags:
        if tag == 'SND':
            num_header += 1
        else:
            break

    to_include = list(range(num_header))
    length = len(prompt_template) + len('\n'.join(rows[:num_header])) + len(get_answer(tags, rows))
    if tags[-1] != "SND":
        to_include.append(len(tags)-1)
        length += len(rows[-1])
    if length > limit:
        return None
    is_non_snd = False
    for index, tag in enumerate(tags[num_header:-1], start=num_header):
        if tag == "SND" and is_non_snd is False:
            continue
        if tag == "AGG" or tag == "SND" or tag == "SEC":
            if length + len(rows[index]) + 1 <= limit:
                is_non_snd = True
                to_include.append(index)
                length += len(rows[index]) + 1
        if tag == "BOD":
            if length + len(rows[index]) <= limit:
                is_non_snd = True
                to_include.append(index)
                length += len(rows[index]) + 1
    index = num_header
    while index < len(tags) - 1:
        if to_include.count(index) != 0:
            index += 1
        elif tags[index] == "SND" and is_non_snd is False:
            index += 1
        elif length + len(rows[index]) <= limit:
            to_include.append(index)
            length += len(rows[index]) + 1
            index += 1
        else:
            break

    to_include = list(set(to_include))
    to_include.sort()
    rows = [rows[i] for i in to_include]
    tags = [tags[i] for i in to_include]

    if len(to_include) <= 2:
        return None

    assert get_answer(tags, rows) == get_answer(ori_tags, rows)
    return {"prompt": prompt_template.format(to_markdown_table(rows)), "answer": get_answer(tags, rows)}


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
    output = prompt_template.format(to_markdown_table(list_of_row_original_table))
    if len(output) + len(get_answer(tags, list_of_row_original_table)) + 2 <= limit:
        return {"prompt": output, "answer": get_answer(tags, list_of_row_original_table)}
    else:
        return extract_from_table(list_of_row_original_table, tags)


def get_feature(feature_file: str):
    # get input
    count = 0
    err_count = 0
    dic = get_full_table()
    with open(save_prefix + 'original_feature/' + feature_file + '.txt', 'r') as f:
        tables = f.readlines()

    # get output
    output = []
    for table in tqdm(tables):
        data = get_output_from_table_one(table, dic)
        if data:
            if len(data['prompt']) + len(data['answer']) > limit + 20:
                err_count += 1
                continue
            output.append(data)
        else:
            count += 1

    print(count)
    print(err_count)
    # write into one json file
    f = jsonlines.open(save_prefix + 'tablesense_dataset_new/' + feature_file + '.json', 'w')
    jsonlines.Writer.write(f, output)


if __name__ == '__main__':
    for feature_file in ['train_row_feature', 'test_263_row_feature']:
        get_feature(feature_file)

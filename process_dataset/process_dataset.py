from tqdm import tqdm
import jsonlines

# SETTINGS
save_prefix = "/spot/v-qinyuxu/"

prompt_template = """<BEGIN_TABLE> {} <END_TABLE> <BEGIN_QUESTION> 1: The number of header rows; 2: Whether there are aggregation rows or multiple non-homogeneous data sections between the header and the second-to-last row(true); 3: If the last row is an aggregation(true) or not(false).
<BEGIN_ANSWER> """

limit = 3200


def to_markdown_table(rows):
    return '\n'.join(rows)

def get_answer(tags):
    num_header = 0
    for tag in tags:
        if tag == "SND" or tag == "HEADER":
            num_header += 1
        else:
            break

    is_homo = True
    for tag in tags[num_header:-1]:
        if tag == "SND" or tag == "AGG" or tag == "SEC":
            is_homo = False
    if tags[num_header:-1].count("BOD") > 1:
        is_homo = False
    is_agg = tags[-1] == "AGG"

    return [num_header, is_homo, is_agg]


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


def part(table_list, index):
    table = '\n'.join(table_list)
    if len(table) > limit:
        new_table = table_list[index]
        is_before = False
        cur_idx_before = index - 1
        cur_idx_after = index + 1
        while len(new_table) < limit and not (cur_idx_after >= len(table_list) and cur_idx_before < 0):
            if cur_idx_after >= len(table_list):
                is_before = True
            if cur_idx_before < 0:
                is_before = False
            if is_before and cur_idx_before >= 0:
                tmp_table = table_list[cur_idx_before] + " \n " + new_table
                if len(prompt_template.format_map({'table': tmp_table, 'row': table_list[index]})) > limit :
                    return new_table
                else:
                    new_table = tmp_table
                cur_idx_before -= 1
            elif not is_before and cur_idx_after < len(table_list):
                tmp_table = new_table + " \n " + table_list[cur_idx_after]
                if len(prompt_template.format_map({'table': tmp_table, 'row': table_list[index]})) > limit :
                    return new_table
                else:
                    new_table = tmp_table
                cur_idx_after += 1
            is_before = not is_before
        return new_table
    else:
        return table


def extract_from_table(rows, tags):
    # for all the rows starting from the header down to the second to last row
    # if BOD exists, then it's necessary to extract at least one BOD and one DAT in this data section
    # if SND exists, then include it
    # if BLA exists, then discard it
    # if AGG exists, then keep it

    num_header = 0
    for tag in tags:
        if tag == 'SND':
            num_header += 1
        else:
            break

    to_include = [len(tags)-1]
    length = 150 - 2 + 14 + len(rows[-1])
    is_skip = False
    for index, tag in enumerate(tags[num_header:-1]):
        if is_skip:
            is_skip = False
            continue
        if tag == "AGG" or tag == "SND":
            if length + len(rows[index]) <= limit:
                to_include.append(index)
                length += len(rows[index])
        if tag == "BOD":
            if index != len(tags)-2 and rows[index+1] == "DAT" and length + len(rows[index]) + len(rows[index+1]) <= limit:
                to_include.append(index)
                to_include.append(index+1)
                length += len(rows[index]) + len(rows[index+1])
                is_skip = True
    index = num_header
    while index < len(tags):
        if to_include.count(index) != 0:
            index += 1
        elif length + len(rows[index]) < limit:
            to_include.append(index)
            length = length + len(rows[index])
        else:
            break

    to_include.sort()
    rows = [rows[i] for i in to_include]
    tags = [tags[i] for i in to_include]
    if len(to_include) < 2:
        with open('check.txt', 'a') as f:
            f.write(prompt_template.format(to_markdown_table(rows)) + '\n' + ','.join(map(lambda x: str(x), get_answer(tags))) + '\n\n')
        return None
    return {"prompt": prompt_template.format(to_markdown_table(rows)), "answer": get_answer(tags)}


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
    if len(output) + len(','.join(map(lambda x: str(x), get_answer(tags)))) + 2 <= limit:
        return {"prompt": output, "answer": get_answer(tags)}
    else:
        return extract_from_table(list_of_row_original_table, tags)


def get_feature(feature_file: str):
    # get input
    count = 0
    dic = get_full_table()
    with open(save_prefix + 'original_feature/' + feature_file + '.txt', 'r') as f:
        tables = f.readlines()

    # get output
    output = []
    for table in tqdm(tables):
        data = get_output_from_table_one(table, dic)
        if data:
            output.append(data)
        else:
            count += 1

    print(count)
    # write into one json file
    f = jsonlines.open(save_prefix + 'tablesense_dataset_new/' + feature_file + '.json', 'w')
    jsonlines.Writer.write(f, output)


if __name__ == '__main__':
    for feature_file in ['train_row_feature', 'test_263_row_feature']:
        get_feature(feature_file)

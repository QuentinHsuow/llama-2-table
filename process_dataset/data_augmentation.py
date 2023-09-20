import random


def delete_snd(data):
    header_number = data['answer']
    to_delete = random.randint(1, header_number-1)
    new_rows = []
    for index, row in enumerate(data['rows']):
        if index < header_number - to_delete or index >= header_number:
            new_rows.append(row)
    header_number -= to_delete
    assert header_number > 0
    return {
        'rows': new_rows,
        'answer': header_number,
    }

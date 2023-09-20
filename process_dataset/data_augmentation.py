import random


def delete_snd(data):
    row_number = data['answer']
    to_delete = random.randint(0, row_number)
    new_rows = []
    for index, row in enumerate(data['rows']):
        if index < row_number - to_delete or index >= row_number:
            new_rows.append(row)
    row_number -= to_delete
    assert row_number > 0
    return {
        'rows': new_rows,
        'answer': row_number,
    }

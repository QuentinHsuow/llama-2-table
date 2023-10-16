def calculate_f1(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1


def get_type(header_num, middle_homo, final_AGG):
    if header_num == 1 and middle_homo is True:
        if final_AGG is True:
            return 2
        else:
            return 1
    else:
        return 3


def calculate_stat(result, index):
    TP = FP = FN = TN = 0
    for data in result:
        res_type = data['res_type']
        ans_type = data['type']
        if res_type == index:
            if ans_type == index:
                TP += 1
            else:
                FP += 1
        else:
            if ans_type == index:
                FN += 1
            else:
                TN += 1
    print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")
    return calculate_f1(TP=TP, FP=FP, FN=FN)

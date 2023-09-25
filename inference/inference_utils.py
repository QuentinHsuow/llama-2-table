def calculate_f1(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1
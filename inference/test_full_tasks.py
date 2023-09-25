import json
from inference_utils import get_type, calculate_stat

def main():
    correct = 0
    with open("saved_result/result.json", 'r') as f:
        result = json.load(f)
    for data in result:
        data['res_type'] = get_type(data['result1'], data['result2'], data['result3'])
        if data['res_type'] == data['type']:
            correct += 1
    print(f"Correct: {correct}, percentage: {correct/len(result)}")
    for index in range(1, 4):
        precision, recall, f1 = calculate_stat(result, index)
        print(f"Type{index}: Precision {precision}, Recall{recall}, f1{f1}")


if __name__ == "__main__":
    main()
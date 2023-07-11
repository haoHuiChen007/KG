# -*- coding : utf-8 -*-
# coding: utf-8
import json


def pro_data(dataset):
    result = []
    sentence = []
    ner = []
    current_index = 0  # 代表word在句子中的索引
    begin_index = 0
    e_type = ""  # 记录实体类型
    n_type = ""
    for line in dataset:
        item = line.split(" ")
        if len(item) != 2:
            one = {"sentence": sentence, "ner": ner}
            result.append(one)
            sentence = []
            ner = []
            current_index = 0
            begin_index = 0
            e_type = ""
            continue
        w, t = item
        if 'B-' in t:
            begin_index = current_index
            e_type = t[2:]
        elif 'O' in t:
            if n_type == 'B' or n_type == 'I':
                ner.append({"index": [i for i in range(begin_index, current_index)], "type": e_type})
        n_type = t[:1]
        current_index += 1
        sentence.append(w)
    return result


if __name__ == "__main__":
    with open(r"D:\py\W2NER-main\W2NER-main\data\dataset\test.test", encoding='unicode_escape') as f:
        all_data = f.read().split("\n")
    new_data = pro_data(all_data)
    with open('data/dataset/test.json', "w", encoding='utf-8') as file:
        text = json.dumps(new_data)
        file.write(text)
import json


def relation_schema(b_type):
    if b_type[0] == 'MAT' and b_type[1] == 'MET':
        return "MAT-MET(e1,e2)"
    elif b_type[1] == 'MAT' and b_type[0] == 'MET':
        return "MAT-MET(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "ATTR":
        return "MAT-ATTR(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "ATTR":
        return "MAT-ATTR(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "CON":
        return "MAT-CON(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "CON":
        return "MAT-CON(e2,e1)"

    elif b_type[0] == "CON" and b_type[1] == "DATA":
        return "CON-DATA(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "DATA":
        return "CON-DATA(e2,e1)"

    elif b_type[0] == "DATA" and b_type[1] == "ATTR":
        return "DATA-ATTR(e1,e2)"
    elif b_type[1] == "DATA" and b_type[0] == "ATTR":
        return "DATA-ATTR(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "APL":
        return "MAT-APL(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "APL":
        return "MAT-APL(e2,e1)"

    elif b_type[0] == "ATTR" and b_type[1] == "APL":
        return "ATTR-APL(e1,e2)"
    elif b_type[1] == "ATTR" and b_type[0] == "APL":
        return "ATTR-APL(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "DSC":
        return "MAT-DSC(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "DSC":
        return "MAT-DSC(e2,e1)"

    elif b_type[0] == "CON" and b_type[1] == "RES":
        return "CON-RES(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "RES":
        return "CON-RES(e2,e1)"
    else:
        return "Other"


# 将bio转为{sentence,ner}形式
def pro_data2dir(dataset):
    result = []
    sentence = []
    ner = []
    current_index = 0  # 代表word在句子中的索引
    begin_index = 0
    e_type = ""  # 记录实体类型
    n_type = ""  # 下一个实体类型
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
            n_type = ""
            continue
        w, t = item
        if 'B' in t:
            begin_index = current_index
            e_type = t[2:]
        elif 'I' not in t:
            if n_type == 'I' or n_type == 'B':
                ner.append({"index": [i for i in range(begin_index, current_index)], "type": e_type})
        n_type = t[:1]
        current_index += 1
        sentence.append(w)
    return result


def pro_data2re(dataset):
    result = []
    index = 0
    for data in dataset:
        ner_len = len(data['ner'])
        if ner_len < 2:
            continue
        for i in range(ner_len):
            for j in range(i + 1, ner_len):
                b_type = [data['ner'][i]['type'], data['ner'][j]['type']]
                relation = relation_schema(b_type)
                if relation == 'Other':
                    continue
                s = []
                s.extend(data['sentence'])
                s.insert(data['ner'][i]['index'][0], '<e1>')
                s.insert(data['ner'][i]['index'][0] + len(data['ner'][i]['index']) + 1, '</e1>')
                s.insert(data['ner'][j]['index'][0] + 2, '<e2>')
                s.insert(data['ner'][j]['index'][0] + len(data['ner'][j]['index']) + 3, '</e2>')
                one = {"id": index, "relation": relation, "sentence": s}
                index += 1
                result.append(one)
    return result


if __name__ == "__main__":
    with open(r"D:\py\KG\torch_bert_bilstm_crf\data\train\train.txt", encoding='unicode_escape') as f:
        all_data = f.read().split("\n")
    # new_data = pro_data2re(pro_data2dir(all_data))
    new_data = pro_data2dir(all_data)
    with open(r'D:\py\KG\relation_Extraction\data\all_12_21_400.json', "w", encoding='utf-8') as file:
        text = json.dumps(new_data)
        file.write(text)

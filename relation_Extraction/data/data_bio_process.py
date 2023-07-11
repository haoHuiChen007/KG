import json


def relation_schema(b_type):
    print(b_type)
    if b_type[0] == 'MAT' and b_type[1] == 'MET':
        return "MAT-MET(e1,e2)"
    elif b_type[1] == 'MAT' and b_type[0] == 'MET':
        return "MAT-MET(e1,e2)"
    elif b_type[0] == "MET" and b_type[1] == "ATTR":
        return "MET-ATTR(e1,e2)"
    elif b_type[1] == "MET" and b_type[0] == "ATTR":
        return "MET-ATTR(e2,e1)"
    elif b_type[0] == "MET" and b_type[1] == "CON":
        return "MAT-CON(e1,e2)"
    elif b_type[1] == "MET" and b_type[0] == "CON":
        return "MAT-CON(e2,e1)"
    elif b_type[0] == "CON" and b_type[1] == "DATA":
        return "CON-DATA(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "DATA":
        return "CON-DATA(e2,e1)"
    elif b_type[0] == "DATA" and b_type[1] == "ATTR":
        return "DATA-ATTR(e1,e2)"
    elif b_type[1] == "DATA" and b_type[0] == "ATTR":
        return "DATA-ATTR(e2,e1)"
    elif b_type[0] == "MET" and b_type[1] == "APL":
        return "MAT-APL(e1,e2)"
    elif b_type[1] == "MET" and b_type[0] == "APL":
        return "MAT-APL(e2,e1)"
    elif b_type[0] == "MET" and b_type[1] == "ATTR":
        return "MAT-ATTR(e1,e2)"
    elif b_type[1] == "MET" and b_type[0] == "ATTR":
        return "MAT-ATTR(e2,e1)"
    elif b_type[0] == "ATTR" and b_type[1] == "APL":
        return "ATTR-APL(e1,e2)"
    elif b_type[1] == "ATTR" and b_type[0] == "APL":
        return "ATTR-APL(e2,e1)"
    elif b_type[0] == "MET" and b_type[1] == "DSC":
        return "MAT-DSC(e1,e2)"
    elif b_type[1] == "MET" and b_type[0] == "DSC":
        return "MAT-DSC(e2,e1)"
    elif b_type[0] == "CON" and b_type[1] == "RES":
        return "CON-RES(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "RES":
        return "CON-RES(e2,e1)"
    else:
        return "Other"


def to_sentence_bio(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        s = list(map(lambda x: x.strip(), f.readlines()))
        for k, i in enumerate(s):
            if i == ". O":
                s[k] = i + "\n"
            if i != ". O" and str(i).endswith(". O"):
                a = i.index(". O")
                s[k] = i[:a] + " O\n" + i[a:] + "\n"
    p = open(output_path, 'w', encoding="utf-8")
    for i in s:
        print(i, file=p)
    p.close()


def per_process(dataset):
    index = 0
    result = []
    sentence = []
    b_type = []
    n_type = "O"
    current_index = 0
    for line in dataset:
        item = line.split(" ")
        # 句子结束
        if len(item) != 2:
            #  判断 实体个数 是否只有两个(第二层过滤)
            if len(b_type) == 2:
                relation = relation_schema(b_type)
                one = {"id": index, "relation": relation, "sentence": sentence}
                result.append(one)
                index += 1
            current_index = 0
            sentence = []
            b_type = []
            continue
        w, t = item
        sentence.append(w)
        if 'B-' in t:
            b_type.append(t[2:])
            insert_t = "<e" + str(len(b_type)) + ">"
            sentence.insert(current_index, insert_t)
            current_index += 1
        elif n_type != 'O' and 'I-' not in t:
            insert_t = "</e" + str(len(b_type)) + ">"
            sentence.insert(current_index, insert_t)
            current_index += 1
        n_type = t[:1]
        current_index += 1
    return result


if __name__ == '__main__':
    to_sentence_bio(r"D:\py\KG\relation_Extraction\data\ner\zyouttrain.txt", r"D:\py\KG\relation_Extraction\data\ner\zyouttrain.txt")
    print("over")
    # with open(r"D:\py\KG\relation_Extraction\data\new\new_all_12_21_400.bio", encoding='unicode_escape') as f:
    #     all_data = f.read().split("\n")
    # new_data = per_process(all_data)
    # with open(r'D:\py\KG\relation_Extraction\data\all_12_21_400.json', "w", encoding='utf-8') as file:
    #     text = json.dumps(new_data)
    #     file.write(text)

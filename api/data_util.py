#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import random
import re
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize


def data2sentence(data_input):
    sentences_output = []
    for sentence_output in sent_tokenize(data_input.replace('\n', '')):
        sentences_output.append(sentence_output)
    return sentences_output


def open_json(save_path):
    with open(save_path) as file:
        text = file.read()
        if text == '':
            return []
        else:
            return json.loads(text)


def relation_schema(b_type):
    if b_type[0] == 'MAT' and b_type[1] == 'MET':
        return "MAT-MET(e1,e2)"
    elif b_type[1] == 'MAT' and b_type[0] == 'MET':
        return "MAT-MET(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "ATTR":
        return "MAT-ATTR(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "ATTR":
        return "MAT-ATTR(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "DATA":
        return "MAT-DATA(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "DATA":
        return "MAT-DATA(e2,e1)"

    elif b_type[0] == "CON" and b_type[1] == "DATA":
        return "CON-DATA(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "DATA":
        return "CON-DATA(e2,e1)"

    # elif b_type[0] == "DATA" and b_type[1] == "ATTR":
    #     return "DATA-ATTR(e1,e2)"
    # elif b_type[1] == "DATA" and b_type[0] == "ATTR":
    #     return "DATA-ATTR(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "APL":
        return "MAT-APL(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "APL":
        return "MAT-APL(e2,e1)"

    # elif b_type[0] == "ATTR" and b_type[1] == "APL":
    #     return "ATTR-APL(e1,e2)"
    # elif b_type[1] == "ATTR" and b_type[0] == "APL":
    #     return "ATTR-APL(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "DSC":
        return "MAT-DSC(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "DSC":
        return "MAT-DSC(e2,e1)"

    # elif b_type[0] == "CON" and b_type[1] == "RES":
    #     return "CON-RES(e1,e2)"
    # elif b_type[1] == "CON" and b_type[0] == "RES":
    #     return "CON-RES(e2,e1)"
    else:
        return "Other"


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


def dict2bio(dataset):
    result = []
    for data in dataset:
        sentence = data["sentence"]
        ner = data["ner"]
        for i, w in enumerate(sentence):
            match = False
            for j, entity in enumerate(ner):
                index = entity["index"]
                type_ = entity["type"]
                if i in index:
                    t = get_bio_tag(index.index(i), type_)
                    result.append((w, t))
                    match = True
                    break
            if not match:
                result.append((w, "O"))
    return result


def get_bio_tag(index, type_):
    if index == 0:
        return "B-" + type_
    else:
        return "I-" + type_


def get_entity_and_type(dataset):
    nodes = []
    pages = []
    for data_ in dataset:
        entity = []
        entity_type = re.split("-|\(|\,", data_["relation"])
        if entity_type[2] == "e2":
            types = entity_type[1::-1]
        else:
            types = entity_type[:2]

        sentence = re.split("<e1>|</e1>|<e2>|</e2>", " ".join(data_["sentence"]))
        for i, s in enumerate(sentence):
            if i == 1 or i == 3:
                entity.append(s)
        sentence = " ".join(data_["sentence"])
        dou_node = [{"name": n, "type": t} for n, t in zip(entity, types)]
        relation = types[0] + "_" + types[1]
        pages.append({"sentence": sentence, "relation": relation, "dou_node": dou_node})
        nodes.extend(dou_node)
    unique_node = list(set([json.dumps(d) for d in nodes]))
    unique_node = [json.loads(d) for d in unique_node]
    return unique_node, pages


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


def check_hypernymy(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is None:
                continue
            # 如果路径相似度为1，则表示两个词是同义词，返回None
            if similarity == 1:
                return None
            # 如果路径相似度大于0.5，则表示两个词有上下位词关系，根据最短路径长度判断方向，并返回结果
            if similarity > 0.5:
                shortest_path = synset1.shortest_path_distance(synset2)
                if shortest_path > 0:
                    return word1, "is a", word2
                else:
                    return word2, "is a", word1
    # 如果没有找到任何上下位词关系，则返回None
    return None


def data_strong():
    # 定义一个包含不同类别实体的词典
    entity_dict = {
        "LOC": ["北京", "上海", "南京", "广州", "深圳"],
        "PER": ["李白", "杜甫", "王维", "白居易", "李清照"],
        "ORG": ["必应", "谷歌", "微软", "苹果", "华为"]
    }

    # 读取原始的bio数据文件
    with open("original_bio.txt") as f:
        lines = f.readlines()

    # 创建一个新的bio数据文件
    with open("new_bio.txt", mode="w") as f:
        # 遍历每一行
        for line in lines:
            # 去掉换行符
            line = line.strip()
            # 如果是空行，直接写入新文件
            if not line:
                f.write("\n")
                continue
            # 将单词和标签分割开来
            word, label = line.split()
            # 如果标签表示一个实体开始（如B-LOC）
            if label.startswith("B-"):
                # 获取实体类别（如LOC）
                entity_type = label.split("-")[1]
                # 从词典中随机选择一个同类别的实体替换单词，并保持标签不变
                new_word = random.choice(entity_dict[entity_type])
                new_line = new_word + "\t" + label + "\n"
            else:
                # 否则保持单词和标签不变
                new_line = word + "\t" + label + "\n"
            # 将新的一行写入新文件
            f.write(new_line)


# 对句子进行分词和命名实体识别，获取实体列表（这里简化为只考虑名词）

if __name__ == '__main__':
    to_sentence_bio(r"D:\py\KG\api\data\train\all.bio", r"D:\py\KG\api\data\train\all.bio")
    # with open(r"data/predict02.txt", encoding='unicode_escape') as f:
    #     all_data = f.read().split("\n")
    # bio = pro_data2re(pro_data2dir(all_data))
    # for word, tag in bio:
    #     print(word, tag)
    # sentence = "LiFePO4/C active materials were synthesized via a modified carbothermal method, with a low raw material cost and comparatively simple synthesis process. Rheological phase technology was introduced to synthesize the precursor, which effectively decreased the calcination temperature and time. The LiFePO4/C composite synthesized at 700 A degrees C for 12 h exhibited an optimal performance, with a specific capacity about 130 mAh g(-1) at 0.2C, and 70 mAh g(-1) at 20C, respectively.It also showed an excellent capacity retention ratio of 96 % after 30 times charge-discharge cycles at 20C. EIS was applied to further analyze the effect of the synthesis process parameters. The as-synthesized LiFePO4/C composite exhibited better high-rate performance as compared to the commercial LiFePO4 product, which implied that the as-synthesized LiFePO4/C composite was a promising candidate used in the batteries for applications in EVs and HEVs."
    # tokens = nltk.word_tokenize(sentence)
    # tags = nltk.pos_tag(tokens)
    # entities = [tag[0] for tag in tags if tag[1] == "NN"]
    #
    # # 遍历实体列表中的每一对实体，调用函数检查是否有上下位词关系，并输出结果
    # for i in range(len(entities)):
    #     for j in range(i + 1, len(entities)):
    #         relation = check_hypernymy(entities[i], entities[j])
    #         if relation is not None:
    #             print(relation)
    with open(r"D:\py\KG\api\data\train\all.bio", encoding='utf-8') as f:
        all_data = f.read().split("\n")

    new_data = pro_data2re(pro_data2dir(all_data))
    with open(r'D:\py\KG\api\data\train\all.json', "w", encoding='utf-8') as file:
        text = json.dumps(new_data)
        file.write(text)

    # to_sentence_bio(r"D:\py\KG\api\data\train\all.bio", r"D:\py\KG\api\data\train\all.bio")
    print("ll")

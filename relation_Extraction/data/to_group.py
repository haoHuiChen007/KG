import json
import re


def get_dataset():
    with open(r"D:\py\KG\relation_Extraction\data\train.json") as file:
        text = file.read()
        if text == '':
            return []
        else:
            return json.loads(text)

def recover(one):
    sentence_ = one['sentence']
    ner_ = re.split("<e1>|</e1>|<e2>|</e2>", " ".join(sentence_))
    relation_ = one['relation']
    if relation_ == 'MAT-MET(e1,e2)':
        return ner_[1], ner_[3], 'method'
    elif relation_ == 'MAT-MET(e2,e1)':
        return ner_[3], ner_[1], 'method'
    elif relation_ == 'MAT-ATTR(e1,e2)':
        return ner_[1], ner_[3], 'attribution'
    elif relation_ == 'MAT-ATTR(e2,e1)':
        return ner_[3], ner_[1], 'attribution'
    elif relation_ == 'MAT-CON(e1,e2)':
        return ner_[1], ner_[3], 'condition'
    elif relation_ == 'MAT-CON(e2,e1)':
        return ner_[3], ner_[1], 'condition'
    elif relation_ == 'CON-DATA(e1,e2)':
        return ner_[1], ner_[3], 'CON-DATA'
    elif relation_ == 'CON-DATA(e2,e1)':
        return ner_[3], ner_[1], 'CON-DATA'
    elif relation_ == 'DATA-ATTR(e1,e2)':
        return ner_[1], ner_[3], 'DATA-ATTR'
    elif relation_ == 'DATA-ATTR(e2,e1)':
        return ner_[3], ner_[1], 'DATA-ATTR'
    elif relation_ == 'MAT-APL(e1,e2)':
        return ner_[1], ner_[3], 'application'
    elif relation_ == 'MAT-APL(e2,e1)':
        return ner_[3], ner_[1], 'application'
    elif relation_ == 'ATTR-APL(e1,e2)':
        return ner_[1], ner_[3], 'ATTR-APL'
    elif relation_ == 'ATTR-APL(e2,e1)':
        return ner_[3], ner_[1], 'ATTR-APL'

    elif relation_ == 'MAT-DSC(e1,e2)':
        return ner_[1], ner_[3], 'description'
    elif relation_ == 'MAT-DSC(e2,e1)':
        return ner_[3], ner_[1], 'description'

    elif relation_ == 'CON-RES(e1,e2)':
        return ner_[1], ner_[3], 'CON-RES'
    elif relation_ == 'CON-RES(e2,e1)':
        return ner_[3], ner_[1], 'CON-RES'
    else:
        return ner_[1], ner_[2], 'Other'

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

    elif b_type[0] == "MAT" and b_type[1] == "DATA":
        return "MAT-DATA(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "DATA":
        return "MAT-DATA(e2,e1)"

    elif b_type[0] == "CON" and b_type[1] == "DATA":
        return "CON-DATA(e1,e2)"
    elif b_type[1] == "CON" and b_type[0] == "DATA":
        return "CON-DATA(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "APL":
        return "MAT-APL(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "APL":
        return "MAT-APL(e2,e1)"

    elif b_type[0] == "MAT" and b_type[1] == "DSC":
        return "MAT-DSC(e1,e2)"
    elif b_type[1] == "MAT" and b_type[0] == "DSC":
        return "MAT-DSC(e2,e1)"

    else:
        return "Other"


def get_json_data(path):
    with open(path) as file:
        text = file.read()
        if text == '':
            return []
        else:
            return json.loads(text)


def save_sort_json_data(path, data):
    with open(path, "w", encoding='utf-8') as file:
        text = json.dumps(data)
        file.write(text)


if __name__ == '__main__':
    org_data = get_json_data(r"D:\DataController\WeChatData\WeChat Files\wxid_rvo3vnlvoqsr12\FileStorage\File\2023-07/other_save.json")
    org_data.sort(key=lambda x: x['id'])
    new_data = [item for item in org_data if item['relation'] != "Other"]
    save_sort_json_data(r"../data/newsave.json", org_data)

    print("over")

    pass

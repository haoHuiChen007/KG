import json

with open("org/train.json") as file:
    text = file.read()
    dataset = json.loads(text)

for i, data in enumerate(dataset):
    data['id'] = i

with open("org/train.json", 'w', encoding='utf-8') as file:
    text = json.dumps(dataset)
    file.write(text)

import os
import torch
from flask import request, make_response, Response, jsonify
from flask import Flask
from transformers import BertTokenizer
from py2neo import Graph
from werkzeug.utils import secure_filename
from data_util import open_json, data2sentence, pro_data2dir

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制为16MB
graph = Graph("bolt://192.168.78.128:7687", auth=("neo4j", "neo4j123456"))


@app.route('/v1/entity_annotation/en', methods=["POST"])
def entity_annotation():
    """
    前端将输入文本传给后台，后台首先将输入文本按
    句拆分，逐句带入实体识别模型进行知识实体的标注，并将每句文本中的知识实体
    及实体类型返回至前端
    :return:
    """

    dataset = request.values.get('dataset')
    if dataset is None:
        response = make_response("输入不能为空")
        response.status_code = 550
        response.content_type = 'application/json'
        return response
    bio_data = []
    index_2_label = open_json('./data/id2label.json')
    tokenizers = BertTokenizer.from_pretrained(os.path.join("..", "matbert-bandgap"))
    model = torch.load(r"D:\py\KG\api\model\BruceBertCrf_chang.pt")
    sentences = data2sentence(dataset)
    for sentence in sentences:
        text = sentence.split(" ")
        text_idx = tokenizers.encode(text, add_special_tokens=True, return_tensors='pt')
        pre = model.forward(text_idx)
        pre = pre[0][1: -1]
        pre = [index_2_label[i] for i in pre]
        bio_data.extend([f"{w} {t}" for w, t in zip(text, pre)])
        bio_data.append('')
    response = make_response(pro_data2dir(bio_data))
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


@app.route('/v1/kn_query_by_type/', methods=["POST"])
def kn_query_by_type():
    """
    用户输入某一类别的知识实体，便可在后台图数据库中转化成相应的查询语句，并
    将查询结果返回至前端进行展示，得到文献中与该知识实体有关联的其他类别的
    实体。
    :return:
    """
    n = request.values.get("n")
    m = request.values.get("m")
    nodes = []
    edges = []
    if n is None:
        n = 1
    if m is None:
        m = 1
    value = request.values.get("value")
    type_ = request.values.get("type")
    print(value, type_)
    cypher = "MATCH (x:%s)-[r*%d..%d]-(y) WHERE x.name = '%s' RETURN x,r,y" % (
        type_, int(n), int(m), value)
    results = graph.run(cypher)
    # 将结果转为前端所需要的形式
    for result in results:
        x = result["x"]
        r = result["r"][0]
        y = result["y"]
        source = {'label': list(x.labels)[0], 'name': x['name']}
        target = {'label': list(y.labels)[0], 'name': y['name']}
        edge = {'source': r.start_node['name'], 'target': r.end_node['name'], 'label': type(r).__name__,
                'from': r['from']}

        # 将节点和关系添加到相应的列表中
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        if edge not in edges:
            edges.append(edge)
    data = {'nodes': nodes, 'edges': edges}
    g = jsonify(data)
    response = make_response(g)
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


@app.route('/v1/get_value_by_type', methods=["GET"])
def get_value_by_type():
    type_ = ["DATA", "APL", "MAT", "MET", "ATTR", "CON", "DSC"]
    # type_ = request.values.get("type")
    values = []
    for t in type_:
        cypher = "MATCH (x:%s) return x" % t
        results = graph.run(cypher)
        nodes = []
        for result in results:
            name = result['x']['name']
            if name not in nodes:
                nodes.append(name)
        values.append(nodes)
    g = jsonify(values)
    response = make_response(g)
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save('./data/')

    else:
        return '不允许上传该类型的文件'


@app.route('/build_kg', methods=['POST'])
def build_kg():
    if request.method == 'POST':
        file = request.files['file']
        dst = os.path.join(os.path.dirname(__file__), file.filename)
        file.save(dst)
        with open(dst, 'r') as f:
            # 读取bio里面的文件内容
            content = f.read().split("\n")

        os.remove(dst)  # 可选，删除临时文件
        if file:
            filename = secure_filename(file.filename)  # 防止恶意文件名
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'File uploaded successfully'


@app.route('/export_txt')
def export_txt():
    data = request.values.get('dataset')
    response = Response(data)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = 'attachment; filename=data.txt'
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)

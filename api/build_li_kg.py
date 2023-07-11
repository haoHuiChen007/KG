import json
from py2neo import Graph, Node
from data_util import pro_data2re, pro_data2dir, get_entity_and_type


def get_nodes_and_relations(dataset):
    new_data = pro_data2re(pro_data2dir(dataset))
    return get_entity_and_type(new_data)


class LiGraph:
    def __init__(self):
        self.g = Graph("bolt://192.168.78.128:7687", auth=("neo4j", "123456neo4j"))

    '''建立节点'''

    def create_node(self, dataset):
        count = 0
        for node in dataset:
            n = Node(node['type'], name=node['name'])
            self.g.create(n)
            count += 1
            print(count, len(dataset))
        return

    def create_relation(self, dataset):
        count = 0
        # 去重处理
        unique_relation = list(set([json.dumps(d) for d in dataset]))
        unique_relation = [json.loads(d) for d in unique_relation]
        for page in unique_relation:
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{from:'%s'}]->(q)" % (
                page['dou_node'][0]['type'], page['dou_node'][1]['type'], page['dou_node'][0]['name'],
                page['dou_node'][1]['name'], page['relation'], page['sentence'])
            try:
                self.g.run(query)
                count += 1
            except Exception as e:
                print(e)
        return


if __name__ == '__main__':
    # 读取预测文件
    with open(r"data/dev.txt", encoding='utf-8') as f:
        all_data = f.read().split("\n")
    nodes, pages = get_nodes_and_relations(all_data)
    handler = LiGraph()
    handler.create_node(nodes)
    handler.create_relation(pages)
    print("over")

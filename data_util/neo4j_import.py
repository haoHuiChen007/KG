# -*- coding: utf-8 -*-
import pandas as pd
from py2neo import Node, Graph, Relationship, NodeMatcher

invoice_data = pd.read_excel('D:/BaiduNetdiskDownload/pandasDemo/Invoice_data_Demo.xls', header=0)


# print("取到的数据：", invoice_data)
class DataToNeo4j(object):

    def __init__(self):
        """建立连接"""
        link = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.graph = link

        # 定义label即节点类型
        self.buy = 'buy'
        self.sell = 'sell'
        self.graph.delete_all()
        self.matcher = NodeMatcher(link)

        # 三引号是注释，官方小例子,帮助理解
        # Node()定义结点，Relationship()定义关系，create()创建结点或关系
        """
        node3 = Node('animal' , name = 'cat')
        node4 = Node('animal' , name = 'dog')  
        node2 = Node('Person' , name = 'Alice')
        node1 = Node('Person' , name = 'Bob')  
        r1 = Relationship(node2 , 'know' , node1)    
        r2 = Relationship(node1 , 'know' , node3) 
        r3 = Relationship(node2 , 'has' , node3) 
        r4 = Relationship(node4 , 'has' , node2)    
        self.graph.create(node1)
        self.graph.create(node2)
        self.graph.create(node3)
        self.graph.create(node4)
        self.graph.create(r1)
        self.graph.create(r2)
        self.graph.create(r3)
        self.graph.create(r4)
        """

    def create_node(self, node_buy_key, node_sell_key):
        """建立节点"""
        for name in node_buy_key:
            buy_node = Node(self.buy, name=name)
            self.graph.create(buy_node)
        for name in node_sell_key:
            sell_node = Node(self.sell, name=name)
            self.graph.create(sell_node)

    def create_relation(self, df_data):
        """建立联系"""
        m = 0
        for m in range(0, len(df_data)):
            try:
                rel = Relationship(
                    self.matcher.match(self.buy).where("_.name=" + "'" + df_data['buy'][m] + "'").first(),
                    df_data['money'][m],
                    self.matcher.match(self.sell).where("_.name=" + "'" + df_data['sell'][m] + "'").first())

                self.graph.create(rel)
            except AttributeError as e:
                print(e, m)


# 从原数据中将需要创建的实体(买方、卖方)节点抽取出来，将所有的数据全部保存到数组中
def data_extraction():
    """节点数据抽取"""

    # 取出所有买方名称到node_buy_key[]
    node_buy_key = []
    for i in range(0, len(invoice_data)):
        node_buy_key.append(invoice_data['购买方名称'][i])

    node_sell_key = []
    for i in range(0, len(invoice_data)):
        node_sell_key.append(invoice_data['销售方名称'][i])

    # 去除重复的买方/卖方名称
    node_buy_key = list(set(node_buy_key))
    node_sell_key = list(set(node_sell_key))

    # 除了第一行，将所有数据按列取出存到node_list_value[]
    node_list_value = []
    for i in range(0, len(invoice_data)):
        for n in range(1, len(invoice_data.columns)):
            node_list_value.append(invoice_data[invoice_data.columns[n]][i])

    # set()去重,list()转化成列表
    node_list_value = list(set(node_list_value))

    # 返回所有去重后的购买方名称，去重后的销售方名称，以及所有数据
    return node_buy_key, node_sell_key, node_list_value


# 将原数据中需要用到的列抽取出来，并且再次拼成excel的样子
def relation_extraction():
    """联系数据抽取"""

    links_dict = {}
    sell_list = []  # 销售方列表
    money_list = []  # 交易额列表
    buy_list = []  # 购买方列表

    # 取列名--“金额”
    # print("*****", invoice_data.columns[19], "********")

    for i in range(0, len(invoice_data)):
        money_list.append(invoice_data[invoice_data.columns[19]][i])  # 将所有金额依次导入
        sell_list.append(invoice_data[invoice_data.columns[10]][i])  # 将所有销售方依次导入
        buy_list.append(invoice_data[invoice_data.columns[6]][i])  # 将所有购买方依次导入

    # 将数据中int类型全部转成string
    sell_list = [str(i) for i in sell_list]
    buy_list = [str(i) for i in buy_list]
    money_list = [str(i) for i in money_list]

    # 整合数据，将三个list整合成一个dict，字典里面存储了多个数组的首地址
    links_dict['buy'] = buy_list
    links_dict['money'] = money_list
    links_dict['sell'] = sell_list

    # 将数据转成DataFrame---类似excel的格式
    df_data = pd.DataFrame(links_dict)
    return df_data


# 实例化
create_data = DataToNeo4j()

# 调用create_data对象的方法创建结点,传参时调用本文件的data_extraction方法

create_data.create_node(data_extraction()[0], data_extraction()[1])
create_data.create_relation(relation_extraction())

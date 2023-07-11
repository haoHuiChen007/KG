import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF


def predict_mask(text_idx):
    mask = [1] * (len(text_idx) + 2)
    return torch.tensor(mask)


class Config:
    """
    配置类
    """

    def __init__(self, dataset):
        # 模型名称
        self.model_name = "BruceBertCrf"
        # 隐藏层大小
        self.hidden_dim = 768
        self.lstm_hidden = 768
        # 定义的分类类别列表
        self.labels_class = [x.strip() for x in open(dataset + "/data/label2id.txt").readlines()]
        # id relation
        self.relations2id = {v: k for k, v in enumerate(self.labels_class)}
        # 类别的数量
        self.class_num = len(self.labels_class)
        # 训练文件路径
        self.train_path = dataset + "/data/ner/3_25_zi" + "/train.txt"
        self.dev_path = dataset + "/data/ner/3_25_zi" + "/dev.txt"
        self.test_path = dataset + "/data/ner/3_25_zi" + "/test.txt"

        # 模型训练保存路径
        self.save_path = dataset + "/model_dir/" + self.model_name + "_chang.pt"
        # 预测文件保存路径
        self.predict_save = dataset + "/data/predict/ner/predict.txt"
        # 是否使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 数据集保存
        self.dataset_pkl = dataset + '/data/dataset.kpl'
        # 迭代词数
        self.epoch = 30
        # 批次
        self.batch_size = 10
        # 是否是测试
        self.is_test = False
        # Bert模型路径
        self.bert_path = os.path.join("matbert-bandgap")

        self.train_log_path = dataset + '/data/log'
        # bert分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 学习率
        self.lr = 1e-4
        # 文本最大长度
        self.max_len = 50
        # 是否全部微调
        self.full_fine_tuning = True
        # 权重衰减：一种用于防止神经网络过拟合的正则化技术
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8


class BertNerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(768, config.lstm_hidden, batch_first=True, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(768 * 2, config.class_num)
        self.crf = CRF(config.class_num, batch_first=True)
        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_index, batch_label=None, mask=None):
        if mask is None:
            bert_out = self.bert(input_ids=batch_index)
        else:
            bert_out = self.bert(input_ids=batch_index, attention_mask=mask)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0 : 字符级-ner , bert_out1 : 篇章级-文本分类
        # pack = nn.utils.rnn.pack_padded_sequence(bert_out, batch_first=True)
        lstm_out, _ = self.lstm(bert_out0)
        pre = self.classifier(lstm_out)
        if batch_label is not None:
            # return self.loss_fun(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            return -self.crf(pre, batch_label)
        else:
            return self.crf.decode(pre)
            # return torch.argmax(pre, dim=-1)

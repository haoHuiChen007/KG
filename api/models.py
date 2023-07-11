import os
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertNerModel(nn.Module):
    def __init__(self, lstm_hidden, class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(os.path.join('..', 'matbert-bandgap'))
        self.lstm = nn.LSTM(768, lstm_hidden, batch_first=True, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(768 * 2, class_num)
        self.crf = CRF(class_num, batch_first=True)

    def forward(self, batch_index, batch_label=None):
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0 : 字符级-ner , bert_out1 : 篇章级-文本分类
        lstm_out, _ = self.lstm(bert_out0)
        pre = self.classifier(lstm_out)
        if batch_label is not None:
            return -self.crf(pre, batch_label)
        else:
            return self.crf.decode(pre)

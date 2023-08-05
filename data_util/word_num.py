# 词频统计
import codecs
import os
import pandas as pd
from nltk import FreqDist

words_path = os.path.join("data", "words_summary.txt")
save_path = os.path.join("data", "words_summary_num.xlsx")


def get_dataset():
    data_set = []
    for line in codecs.open(words_path, 'r', encoding='utf-8'):
        data_set.append(line.strip('\r').strip('\n').strip('\t').split('\r')[0])
    return data_set


data_set = get_dataset()

freq = FreqDist(data_set)
resdf = pd.DataFrame()
resdf["关键字"] = freq.keys()
resdf["词频"] = freq.values()
resdf.to_excel(save_path, index=False)
print("over")

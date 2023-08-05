import codecs
import operator
import pandas as pd
from nltk.text import TextCollection


def TF_IDF(Corpus, WordList):
    """
        计算文本中每个单词的TF-IDF值
        :param Corpus: 语料库
        :param WordList: 目标文本使用上一步的Split_Word(Text)函数处理得到的词表
        :return: TF_IDF_Dict: 文本中每一个单词及其TF-IDF值构成的字典
    """
    TF_IDF_Dict = {}
    for Word in WordList:
        TF_IDF_Dict[Word] = Corpus.tf_idf(Word, WordList)  # 使用nltk自带的TF-IDF函数对词表中每个词计算其TF-IDF值
    return TF_IDF_Dict


if __name__ == '__main__':
    df = pd.read_excel(r"D:\py\KG\data_util\data\result_lines.xlsx")
    corpus = list(df[df.columns[0]].values)
    Corpus = TextCollection(corpus)  # 使用词表构建语料库


    def get_word():
        words = []
        for line in codecs.open(r'D:\py\KG\data_util\data\words_summary.txt', 'r', encoding='utf-8'):
            words.append(line.strip('\n').strip('\r').strip('\t').split('\r')[0])
        return words


    # for i in range(5):
    #     keys = []
    #     values = []
    #     for item in sorted(TF_IDF(Corpus, [a_ for a_ in list(df[df.columns[i]].values) if a_ == a_]).items(),
    #                        key=operator.itemgetter(1), reverse=True):
    #         (key, value) = item
    #         keys.append(key)
    #         values.append(value)
    #     res = pd.DataFrame()
    keys = []
    values = []
    for item in sorted(TF_IDF(Corpus, [a_ for a_ in get_word() if a_ == a_]).items(),
                       key=operator.itemgetter(1), reverse=True):
        (key, value) = item
        keys.append(key)
        values.append(value)
    res = pd.DataFrame()
    res["关键字"] = keys
    res["tf值"] = values
    res.to_excel("./data/total_tf_idf.xlsx", index=False)

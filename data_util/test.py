import math

import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
from nltk import sent_tokenize
import pandas as pd
from tqdm import tqdm
from data_util.test01 import deep_clean_words
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

words_path = os.path.join('data', 'words_1.txt')
stop_words_path = os.path.join('data', 'stopwords.txt')
rm_stop_words_path = os.path.join('data', 'remove_stop_word.txt')
freq_stat_words_path = os.path.join('data', 'freq_stat_word.txt')
words_set_path = os.path.join("data", "words_num.xlsx")


class TF_IWF:
    '''
    tf-iwf 算法
    '''

    def __init__(self, lines):
        self.iwf = dict()
        self.median_iwf = 0
        self.__build_iwf(lines)

    def __get_tf(self, strs):
        tf_dict = {}
        line_words = strs.split(" ")
        total_word_line = len(line_words)
        for word in line_words:
            if word not in tf_dict:
                tf_dict[word] = 1
            else:
                tf_dict[word] = tf_dict[word] + 1
        for k, v in tf_dict.items():
            tf_dict[k] = v / total_word_line
        return tf_dict

    def __build_iwf(self, lines):

        for line in lines:
            line_words = line.split(" ")
            for word in line_words:
                if word not in self.iwf:
                    self.iwf[word] = 1
                else:
                    self.iwf[word] = self.iwf[word] + 1
        total_word_lines = len(self.iwf.values())
        values = []
        for k, v in self.iwf.items():
            self.iwf[k] = math.log(total_word_lines / v, 10)
            values.append(math.log(total_word_lines / v, 10))
        self.median_iwf = np.median(values)

    def get_tfiwf(self, strs):
        result = dict()
        tf_dict = self.__get_tf(strs)
        line_words = strs.split(" ")
        for word in line_words:
            if word not in self.iwf.keys():
                result[word] = tf_dict[word] * self.median_iwf
            else:
                result[word] = tf_dict[word] * self.iwf[word]
        return result


def get_data():
    data_all_info = pd.read_excel(r"D:\py\KG\data_util\data\清洗后的论文（最终版）.xlsx")
    return list(data_all_info[data_all_info.columns[0]].values), list(
        data_all_info[data_all_info.columns[1]].values), list(data_all_info[data_all_info.columns[2]].values)


def data2sentence(data_input):
    sentences_output = []
    for sentence_output in sent_tokenize(data_input.replace('\n', '')):
        sentences_output.append(sentence_output)
    return sentences_output


def sentence2word(sentence_input):
    token_word = word_tokenize(sentence_input)  # 分词
    token_words = pos_tag(token_word)  # 词性标注
    # 词性归一化
    words_lematizer = []
    wordnet_lematizer = WordNetLemmatizer()
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        # elif tag.startswith('JJ'):
        #     word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        # elif tag.startswith('R'):
        #     word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            # word_lematizer = wordnet_lematizer.lemmatize(word)
            continue
        words_lematizer.append(word_lematizer)
    cleaned_words = [word for word in words_lematizer if word not in stopwords.words('english')]
    # 去除特殊字符
    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
                  '^', '{', '}', '<', '>', '\'', '\'s', '+/-', '/', '--', '°c', '``', '►', '·', '×', '”', '||', '..',
                  '//', '`']
    words_list = [word for word in cleaned_words if word not in characters]
    # 大小写转换
    # 为防止同一个单词同时存在大小写而算作两个单词的情况，还需要统一单词大小写（此处统一为小写）。
    return [x.lower() for x in words_list]


def remove_stop_word(dataset, stop_words):
    stop_words = set(stop_words)
    all_words = [word for word in dataset if len(word) > 1 and word not in stop_words]
    print(len(all_words), all_words[:20])
    pd.DataFrame(all_words).to_csv('data/remove_stop_word.txt', index=False)


def show_tfidf(tfidf, vocab, filename):
    # [n_doc, n_vocab]
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())  # 热图
    plt.xticks(np.arange(tfidf.shape[1]), vocab, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0] + 1), fontsize=6)
    plt.tight_layout()
    plt.savefig("%s.png" % filename, format="png", dpi=500)
    plt.show()


if __name__ == '__main__':
    lines = []
    save_path = os.path.join(".", "data", "tf-iwf-words.txt")
    words = []
    summary = []
    article_titles, keywords, abstracts = get_data()
    print("摘要句子转单词")
    for i, data in enumerate(tqdm(abstracts)):
        sentences = data2sentence(data)
        for sentence in sentences:
            words = sentence2word(sentence)
        summary.append(words)
    print("关键词句子转单词")
    for i, data in enumerate(tqdm(keywords)):
        summary[i].extend(sentence2word(data))
    print("主题转单词")
    for i, data in enumerate(tqdm(article_titles)):
        summary[i].extend(sentence2word(data))
    # sigal_words = []
    # for ws in words:
    #     for w in ws:
    #         sigal_words.append(w)
    # pd.DataFrame(sigal_words).to_csv('data/words.txt')
    clean_words = []
    for i, s in enumerate(summary):
        summary[i] = deep_clean_words(s)

    # for s in summary:
    #     line = " ".join(s)
    #     lines.append(line)
    # vectorizer = TfidfVectorizer()
    # tf_idf = vectorizer.fit_transform(lines)
    # df = pd.read_excel(r"D:\py\KG\data_util\data\words_label.xlsx")
    # words_1 = list(df[df.columns[0]].values)
    # words_2 = list(df[df.columns[1]].values)
    # words_3 = list(df[df.columns[2]].values)
    # words_4 = list(df[df.columns[3]].values)
    # words_5 = list(df[df.columns[4]].values)
    # tfiwf = TF_IWF(lines)
    # excel = pd.read_excel(words_set_path)
    # words_data = excel[excel.columns[0]].values
    # line_words = " ".join(words_data)
    # result = tfiwf.get_tfiwf(line_words)
    # resdf = pd.DataFrame()
    # resdf["关键字"] = result.keys()
    # resdf["词频"] = result.values()
    # resdf.to_excel("data/finally_words.xlsx", index=False)
    # i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
    # dense_tfidf = tf_idf.todense()  # 转换为矩阵
    # show_tfidf(dense_tfidf, [i2v[i] for i in range(dense_tfidf.shape[1])], "tfidf_sklearn_matrix")
    print("over")

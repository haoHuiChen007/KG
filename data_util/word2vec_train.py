import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
from nltk import sent_tokenize
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from data_util.test01 import deep_clean_words
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
# from sklearn.datasets._samples_generator import make_blobs
from sklearn import metrics

words_path = os.path.join('./', 'data', 'words_1.txt')
stop_words_path = os.path.join('./', 'data', 'stopwords.txt')
rm_stop_words_path = os.path.join('./', 'data', 'remove_stop_word.txt')
freq_stat_words_path = os.path.join('./', 'data', 'freq_stat_word.txt')
words_set_path = os.path.join("./", "data", "words_num.xlsx")
model_save_path = os.path.join("./", "data", "embedding_word2vec1.model")
model_save_info_path = os.path.join("./", "data", "embedding_word2vec_info1.xlsx")
vec2words_path = os.path.join("./", "data", "vec2words1.xlsx")


def get_data():
    data_all_info = pd.read_excel(r"D:\py\KG\material\material\DataPrerocessing\附件\清洗后的论文（最终版）.xlsx")
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
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
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


if __name__ == '__main__':
    lines = []
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
    clean_words = []
    for i, s in enumerate(summary):
        summary[i] = deep_clean_words(s)
    model = Word2Vec(summary, vector_size=80, window=5, min_count=2, sg=0, hs=1)
    keys = model.wv.index_to_key
    res = [(key, model.wv[key]) for key in keys]

    # 保存模型
    model.save(model_save_path)
    # 保存词与其对应的向量
    dp = pd.DataFrame(res)
    dp.columns = ['关键字', '词向量']
    dp.to_excel(model_save_info_path, index=False)
    # 将词向量投影到二维空间
    rawWordVec = []
    word2ind = {}
    print("压缩为二维词向量")
    for i, w in enumerate(model.wv.index_to_key):  # index_to_key 序号,词语
        rawWordVec.append(model.wv[w])  # 词向量
        word2ind[w] = i  # {词语:序号}
    rawWordVec = np.array(rawWordVec)
    vec2words = pd.DataFrame(rawWordVec)
    vec2words.to_excel(vec2words_path)

    X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
    # 绘制星空图
    # 绘制所有单词向量的二维空间投影
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    ax.set_facecolor('white')
    ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')
    plt.savefig("./data/fig011.png")
    # 可视化k-means聚类结果
    for index, k in enumerate((2, 3, 4, 5)):
        plt.subplot(2, 2, index + 1)
        y_pred = MiniBatchKMeans(n_clusters=k, batch_size=200, random_state=9).fit_predict(X_reduced)
        score = metrics.calinski_harabasz_score(X_reduced, y_pred)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score:%.2f' % (k, score)), transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
    # 显示图像
    plt.savefig('./data/fig021.png')
    plt.show()
    #
    # for k in range(10):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(rawWordVec)
    #     meandistortions.append(sum(np.min(cdist(rawWordVec,kmeans.cluster_centers_,"euclidean"),axis=1))/rawWordVec.shape[0])
    silhouette_all = []
    y_labels = []
    res = []
    for k in range(2, 10):
        kmeans_model = KMeans(n_clusters=k, random_state=9).fit(X_reduced)
        labels = kmeans_model.labels_
        a = metrics.silhouette_score(X_reduced, labels, metric='euclidean')
        silhouette_all.append(a)
        y_labels.append(labels.tolist())
        # print(a)
        print('这个是k={}次时的轮廓系数：'.format(k), a)

    for index, value in enumerate(tqdm(keys)):
        a = [value] + [y[index] for y in y_labels]
        res.append(tuple(a))
    resdf = pd.DataFrame(res)
    resdf.columns = ['关键字'] + ['n_clusters{}'.format(i) for i in range(2, 10)]
    resdf.to_excel("./data/result_labels.xlsx", index=False)
    print(model.wv.most_similar("cathode", topn=20))

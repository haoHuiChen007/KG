import warnings
from gensim import corpora, models
import pyLDAvis.gensim_models
import codecs
from gensim.test.utils import common_corpus
from gensim.models.coherencemodel import CoherenceModel
import jieba.posseg as pseg
warnings.filterwarnings('ignore')

# deep_clean_words_path = r'D:\py\KG\data_util\data\deep_clean_word_finally1.txt'
#
#
# def get_deep_clean_word():
#     deep_clean_words = []
#     for line in codecs.open(deep_clean_words_path, 'r', encoding='utf-8'):
#         deep_clean_words.append(line.strip('\n').strip('\r').strip('\t').split('\r')[0])
#     return deep_clean_words


# def get_dataset():
#     df = pd.read_excel(r"D:\py\KG\data_util\data\result_labels.xlsx")
#     return list(df[df.columns[0]].values)


# 根据文本列表创建一个语料库，每个词与一个整型索引值对应
# df = pd.read_excel(r"D:\py\KG\data_util\data\words.xlsx")
# deep_clean_word = list(df[df.columns[0]].values)


def get_dataset():
    data_set = []
    for line in codecs.open('D:\py\KG\data_util\data\words_summary.txt', 'r', encoding='utf-8'):
        data_set.append(line.strip('\r').strip('\n').strip('\t').split('\r')[0])
    return data_set


deep_clean_word = get_dataset()
word_list = []
for i in range(len(deep_clean_word)):
    train1 = []
    train1.append(deep_clean_word[i])
    word_list.append(train1)
word_dict = corpora.Dictionary(word_list)

# 词频统计，转化成空间向量格式
corpus_list = [word_dict.doc2bow(text) for text in word_list]

model_list = []
perplexity = []  # 困惑度
coherence_values = []  # 一致性

for num_topics in range(2, 12, 1):
    lda_model = models.LdaModel(corpus=corpus_list, id2word=word_dict, random_state=1, num_topics=num_topics, passes=20,
                                alpha='auto')
    model_list.append(lda_model)  # 不同主题个数下的lda模型

    # 模型对应的困惑度（越低越好）
    perplexity_values = lda_model.log_perplexity(corpus_list)
    print('第 %d 个主题的Perplexity为: ' % (num_topics - 1), round(perplexity_values, 3))
    perplexity.append(round(perplexity_values, 3))
    # 模型对应的一致性（越高越好）
    coherencemodel = CoherenceModel(model=lda_model, corpus=common_corpus, coherence='u_mass')
    coherence_values.append(round(coherencemodel.get_coherence(), 3))
    print('第 %d 个主题的Coherence为: ' % (num_topics - 1), round(coherencemodel.get_coherence(), 3))
print('最大的Coherence为：' + str(max(coherence_values)))
for i in range(len(coherence_values)):
    if coherence_values[i] == max(coherence_values):
        print('对应的主题个数为：' + str(i + 2))

lda = models.LdaModel(corpus=corpus_list, id2word=word_dict, random_state=1, num_topics=7, passes=20, alpha='auto')

d = pyLDAvis.gensim_models.prepare(lda, corpus_list, word_dict, mds='pcoa', sort_topics=True)

pyLDAvis.save_html(d, 'lda_show_2.html')  # 将结果保存为html文件

# 展示在notebook的output cell中
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda, corpus_list, word_dict)

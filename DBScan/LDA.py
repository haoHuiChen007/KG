from pprint import pprint
import gensim
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models import Word2Vec
from nltk import corpus, sent_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.gensim_models



np.random.seed(0)

# 构建数据
model = Word2Vec.load("../embedding/embedding.model")
words = model.wv.index_to_key

df = pd.read_excel("../data/tfidf_all.xlsx")
word = list(df[df.columns[0]].values)

keys = []
train_y = []
for i in tqdm(word):
    if i in words:
        keys.append(i)
        train_y.append(model.wv[i])

print("关键字数目{}".format(len(keys)))


# def print_top_words(model, feature_names, n_top_words):
#     tword = []
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic #%d:" % topic_idx)
#         topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
#         tword.append(topic_w)
#         print(topic_w)
#     return tword
#
#
# n_features = 1000  # 提取1000个特征词语
# tf_vectorizer = CountVectorizer(strip_accents='unicode',
#                                 max_features=n_features,
#                                 stop_words='english',
#                                 max_df=0.5,
#                                 min_df=10)
# tf = tf_vectorizer.fit_transform(keys.content_cutted)
#
# n_topics = 8
# lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
#                                 learning_method='batch',
#                                 learning_offset=50,
#                                 #                                 doc_topic_prior=0.1,
#                                 #                                 topic_word_prior=0.01,
#                                 random_state=0)
# lda.fit(tf)

lda = LatentDirichletAllocation(n_components=3, learning_offset=50, random_state=0,n_jobs=-1,max_iter=1000).fit_transform(train_y)

pyLDAvis.enable_notebook()

def lda_vis():
    #dictionary = gensim.corpora.Dictionary.load('lda.dict')
    #corpus = gensim.corpora.MmCorpus('lda.mm')
    #lda = models.ldamodel.LdaModel.load('lda.lda')
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, corpora.Dictionary(keys))
    pyLDAvis.save_html(vis, r'lda001.html')
    return 0

lda_vis()


# Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=4,
#                                             random_state=100,
#                                             update_every=1,
#                                             chunksize=100,
#                                             passes=10,
#                                             alpha='auto',
#                                             per_word_topics=True)
#
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]
page_content = "Vanadium pentoxide (V2O5) offers high capacity and energy density as a cathode candidate for lithium-ion batteries (LIBs)." \
               " Unfortunately, its practical utilization is intrinsically handicapped by the low conductivity, poor electrode kinetics, and lattice instability." \
               " In this study, the synergistical optimization protocol has been proposed in the conjunction of interstitial Ca incorporation and organic vanadate surface protection." \
               " It is revealed that regulating Ca occupation in the body phase at a relatively low concentration can effectively expand the layer distance of alpha-V2O5, which facilitates the intercalation access for Li-ion insertion. " \
               "On the other hand, organometallics are first applied as the protective layer to stabilize the electrode interface during cycling. The optimized coating layer, vanadium oxy-acetylacetonate (VO(acac)(2)), plays an important role to generate a more inorganic component (LiF) within the solid electrolyte interface, contributing to the protection of the Ca-incorporated V2O5 electrode." \
               " As a result, the optimized Ca0.05V2O5/VO(acac)(2) hybrid electrode exhibits much improved capacity utilization, rate capability, and cycling stability, delivering capacity as high as 297 mAh g(-1) for full LIBs. The first-principle computations reveal the lattice change caused by the Ca incorporation, further confirming the lattice advantage of Ca0.05V2O5/VO(acac)(2) with respect to Li-ion intercalation."
for i in sent_tokenize(page_content.replace('\n', '')):
    print([i])

import codecs

import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

words_set_path = os.path.join("data", "deep_clean_word_finally1.txt")

# excel = pd.read_excel(words_set_path)
# print(excel[excel.columns[0]].values)

# df = pd.read_excel(r"D:\py\KG\data_util\data\result_lines.xlsx")
# corpus = list(df[df.columns[0]].values)
# vectorizer = CountVectorizer()
#
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# res = pd.DataFrame()
# res["排序"] = list(vectorizer.vocabulary_.values())
# res["词"] = list(vectorizer.vocabulary_.keys())
# res.to_excel("./data/result_tf_idf.xlsx")
#
# for i in range(5):
#     print(i)
from gensim.models import Word2Vec

model = Word2Vec.load(r"D:\py\KG\data_util\data\embedding_word2vec1.model")


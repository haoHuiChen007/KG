import nltk
from nltk.corpus import wordnet

def extract_relations(text):
    relations = []
    # 使用 nltk 分词
    words = nltk.word_tokenize(text)
    for i, word in enumerate(words):
        # 对于每个单词，找到它的 WordNet 同义词
        synsets = wordnet.synsets(word)
        for synset in synsets:
            # 对于每个同义词，找到它的定义
            definition = synset.definition()
            # 把定义中出现的单词作为关系的一端
            definition_words = nltk.word_tokenize(definition)
            for definition_word in definition_words:
                relations.append((word, definition_word))
    return relations

relations = extract_relations("The cat sat on the mat")
print(relations)

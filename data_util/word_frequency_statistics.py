import codecs
import os
import re
import pandas as pd
from tqdm import tqdm

words_path = os.path.join('data', 'words_1.txt')
stop_words_path = os.path.join('data', 'stopwords.txt')
rm_stop_words_path = os.path.join('data', 'remove_stop_word.txt')
freq_stat_words_path = os.path.join('data', 'freq_stat_word.txt')
clean_words_path = os.path.join('data', 'words_1.txt')


def get_dataset():
    data_set = []
    for line in codecs.open(words_path, 'r', encoding='utf-8'):
        data_set.append(line.strip('\r').strip('\n').strip('\t').split('\r')[0])
    return data_set
    # return list(pd.read_csv('data/words_1.txt', sep='\t'))


def get_clean_words():
    clean_words = []
    for line in codecs.open(clean_words_path, 'r', encoding='utf-8'):
        clean_words.append(line.strip('\r').strip('\n').strip('\t').split('\r')[0])
    return clean_words


def get_stop_word():
    stop_words = []
    for line in codecs.open(stop_words_path, 'r', encoding='utf-8'):
        stop_words.append(line.strip('\n').strip('\r').strip('\t').split('\r')[0])
    return stop_words
    # return codecs.open('data/stopwords.txt', 'r', encoding='utf-8')
    # return list(pd.read_csv('data/stopwords.txt', sep='\t'))


def get_rm_stop_word():
    stop_words = []
    for line in codecs.open(rm_stop_words_path, 'r', encoding='utf-8'):
        stop_words.append(line.strip('\n').strip('\r').strip('\t').split('\r')[0])
    return stop_words


def remove_stop_word(dataset, stop_words):
    stop_words = set(stop_words)
    all_words = [word for word in dataset if len(word) > 1 and word not in stop_words]
    return all_words
    # pd.DataFrame(all_words).to_csv('data/deep_clean_word_finally.txt', index=False)


def freq_stat():
    data_all_info = get_rm_stop_word()
    # data_all_info = pd.read_csv("D:/py/KG/data_util/data/remove_stop_word.txt", sep='\t')
    # all_words = list(data_all_info[data_all_info.columns[0]].values)
    all_words = set(data_all_info)
    freq_stat_words = pd.Series(all_words).value_counts()
    # pd.DataFrame(freq_stat_words).to_csv('data/freq_stat_word.txt')
    pd.DataFrame(all_words).to_csv('data/not_repeat_words.txt', index=False)


def rm_noise_word():
    freq_stat_words = []
    for line in codecs.open(freq_stat_words_path, 'r', encoding='utf-8'):
        word = line.strip('\n').strip('\r').strip('\t').split('\r')[0].split(',')[0]
        if len(re.findall('\d', word)) == 0 and len(re.findall('=', word)) == 0:
            freq_stat_words.append(word)
    pd.DataFrame(freq_stat_words).to_csv('data/clean_words.txt', index=False)


def deep_clean_words(clean_words):
    deep_clean_ws = []
    deep_clean_ws1 = []
    deep_clean_ws2 = []
    for w in tqdm(clean_words):
        if len(re.findall('\d', w)) == 0 and len(re.findall('=', w)) == 0:
            deep_clean_ws2.append(w)
    for w in tqdm(deep_clean_ws2):
        w = w.split('/')
        deep_clean_ws = deep_clean_ws + w
    for w in tqdm(deep_clean_ws):
        w = w.split('.')
        deep_clean_ws1 = deep_clean_ws1 + w
    stop_words = get_stop_word()
    stop_words = stop_words + ['<', '>', '\'', '\'s', '+/-', '/', '--', '°c', '``', '►', '·', '×', '”', '||', '..',
                               '//', '`']
    return remove_stop_word(deep_clean_ws1, stop_words)


if __name__ == '__main__':
    data = get_dataset()
    stop_words = get_stop_word()
    remove_stop_word(data, stop_words)
    freq_stat()
    rm_noise_word()
    deep_clean_words()
    pass
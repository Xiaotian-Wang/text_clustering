import json
import numpy as np
import jieba
from collections import Counter
from itertools import chain
from zhon.hanzi import punctuation as zh_punctuation
from string import punctuation as eng_punctuation
import re

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
dataset = [item for item in data if item['code'][:2] == '20']
cate_list = [item['code'][:2] for item in data]


documents = [item['text'] for item in dataset]

# Filter the data with '参见'
documents = [item for item in documents if '参见' not in item]
true_labels = [item['cate'] for item in dataset if '参见' not in item['text']]
names = [item['name'] for item in dataset if '参见' not in item['text']]

names = np.array(names)
documents = np.array(documents)
true_labels = np.array(true_labels)


def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    #
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return stopword_list


# clean the stopwords in a wordlist
def clean_stopword(word_list, stopword_list):
    result = []
    for w in word_list:
        if w not in stopword_list:
            result.append(w)
    return result


stopwords = get_stopword_list('stopwords.txt')
stopwords += list(zh_punctuation)
stopwords += list(eng_punctuation)


def word_matrix(documents):
    '''计算词频矩阵'''
    # lower all letters
    docs = [d.lower() for d in documents]
    # word seg
    docs = [clean_stopword(jieba.lcut(d),stopwords) for d in docs]
    # get all words
    words = list(set(chain(*docs)))

    # index all the words
    dictionary = dict(zip(words, range(len(words))))

    # a new empty matrix
    matrix = np.zeros((len(words), len(docs)))
    # 逐个文档统计词频
    for col, d in enumerate(docs):  # col 表示矩阵第几列，d表示第几个文档。
        # 统计词频
        count = Counter(d)  # 其实是个词典，词典元素为：{单词：次数}。
        for word in count:
            # 用word的id表示word在矩阵中的行数，该文档表示列数。
            id = dictionary[word]
            # 把词频赋值给矩阵
            matrix[id, col] = count[word]
    return matrix, dictionary

def get_first_sents(text, split_sent_pattern, n_sents=3):
    sents = re.split(split_sent_pattern, text)
    sents = sents[:n_sents]
    sents = '。'.join(sents)
    return sents

documents = np.array([get_first_sents(item, split_sent_pattern,100) for item in documents])


matrix, dictionary = word_matrix(documents)
# print(matrix, '\n', dictionary)

X = np.transpose(matrix)
np.save('X.npy',X)
np.save('documents.npy', documents)
np.save('true_labels.npy', true_labels)
np.save('names.npy', names)

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer


transformer = TfidfTransformer()
X = transformer.fit_transform(X=X)
X = X.toarray()

from sklearn.cluster import KMeans

db = KMeans(n_clusters=7).fit(X)
db = DBSCAN(min_samples=3, eps=1).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
#print(
#    "Adjusted Mutual Information: %0.3f"
#    % metrics.adjusted_mutual_info_score(labels_true, labels)
#)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]

    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

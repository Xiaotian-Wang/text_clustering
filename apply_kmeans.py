import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from collections import Counter
from zhon.hanzi import punctuation as zh_punctuation
from string import punctuation as eng_punctuation
import jieba
from itertools import chain
from transformers import BertTokenizer
from transformers import BertModel
import torch


split_sent_pattern = "[；;.。！!？?\n]"

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

def word_matrix(documents, stopwords):
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

def apply_kmeans(X, n_clusters):
    result = KMeans(n_clusters=n_clusters).fit(X)
    return result


def get_first_sents(text, split_sent_pattern, n_sents=3):
    sents = re.split(split_sent_pattern, text)
    sents = sents[:n_sents]
    sents = '。'.join(sents)
    return sents


class BertEncoder(object):
    def __init__(self):
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.device_cpu = torch.device('cpu')

    def encode(self, text):
        input = self.tokenizer(text)
        input_ids = torch.tensor(input.get('input_ids')).to(self.device)
        token_type_ids = torch.tensor(input.get('token_type_ids')).to(self.device)
        attention_mask = torch.tensor(input.get('attention_mask')).to(self.device)
        result = self.encoder(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)).pooler_output.squeeze()
        # result.requires_grad = False
        result = result.to(self.device_cpu).detach().numpy()
        return result

if __name__ == '__main__':

    documents = np.load('documents.npy')
    documents = np.array([get_first_sents(item, split_sent_pattern, 3) for item in documents])

        # print(matrix, '\n', dictionary)

    # Extract the TF-IDF features
    stopwords = get_stopword_list('stopwords.txt')
    stopwords += list(zh_punctuation)
    stopwords += list(eng_punctuation)
    matrix, dictionary = word_matrix(documents, stopwords=stopwords)
    X = np.transpose(matrix)
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X=X)
    X = X.toarray()

    # encoder = BertEncoder()
    # X = np.array([encoder.encode(item) for item in documents])

    true_labels = np.load('true_labels.npy')
    names = np.load('names.npy')
    result = apply_kmeans(X, n_clusters=4)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X)

    # Visualization
    unique_labels = set(result.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    ax1 = plt.axes(projection='3d')
    for i in range(len(result.labels_)):
        ax1.scatter3D(pca_result[i][0],pca_result[i][1],pca_result[i][2],'o', color=colors[result.labels_[i]])#,markeredgecolor="k",)

    plt.show()

    np.dstack((names, result.labels_, true_labels, np.array([len(item) for item in documents])))

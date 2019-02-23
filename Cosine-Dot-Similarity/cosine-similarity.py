import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

doc_list = []
words_list = []

DOCUMENT_PATH = "Data/docs"
QUERY_PATH = "Data/queries"

def get_unique_words():
    global doc_list
    global words_list
    for filename in os.listdir(DOCUMENT_PATH):
        temp = []
        doc_list.append(filename)
        f = open(DOCUMENT_PATH + "/" + filename, 'r')
        lines = f.readlines()
        for line in lines:
            temp.append(line.strip())
        words_list.append(temp)

    # merge list of arrays to an array
    words = []
    for index, wl in enumerate(words_list):
        words += wl

    return set(words)


def get_document_term_matrix(vocabulary):
    wordDict_list = []
    for wl in words_list:
        wordDict = dict.fromkeys(vocabulary, 0)
        for word in wl:
            wordDict[word] += 1
        wordDict_list.append(wordDict)

    df = pd.DataFrame(wordDict_list)

    # store document-term-matrix to a csv
    df.to_csv('document_term.csv', index=None, header=True)
    return df


def get_query_term_matrix(filename, vocabulary):
    f = open(QUERY_PATH + "/" + filename, 'r')
    query_wordDict = dict.fromkeys(vocabulary, 0)

    for word in f.readlines():
        w = word.strip()
        if w in query_wordDict.keys():
            query_wordDict[w] += 1

    df = pd.DataFrame([query_wordDict])

    # store query-term-matrix to a csv
    df.to_csv('query_term.csv', index=None, header=True)
    return df


def compute_cosine_similarity(d, q):
    cosine_similarity = []
    for i in d:
        i = i.reshape(i.shape[0], 1)
        cosine_similarity.append(np.dot(i.T, q.T) / (norm(i) * norm(q)))
    t = zip(cosine_similarity, doc_list)

    sorted_res = sorted(t, reverse=True)

    cosine_similarity, document = zip(*list(sorted_res))

    cosine_similarity_list = list(cosine_similarity)
    document_list = list(document)

    print("=========Top 10 most similar documents=========")
    print("Document", "Cosine-Similarity value")
    for doc, sim in zip(document_list[:10], cosine_similarity_list[:10]):
        print(doc, sim[0][0])


def main():
    vocabulary = get_unique_words()
    document_term_matrix = get_document_term_matrix(vocabulary)

    # execute for each query
    for filename in os.listdir(QUERY_PATH):
        print()
        print("RESULT for Query:{}".format(filename))
        query_term_matrix = get_query_term_matrix(filename, vocabulary)
        compute_cosine_similarity(document_term_matrix.values, query_term_matrix.values)


if __name__ == '__main__':
    main()

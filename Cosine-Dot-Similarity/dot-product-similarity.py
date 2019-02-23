import os
import pandas as pd
import numpy as np

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
    for wl in words_list:
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


def compute_dot_product_similarity(d, q):
    dot_similarity = np.dot(d, q.T)

    t = zip(dot_similarity, doc_list)

    sorted_res = sorted(t, reverse=True)

    dot_similarity, document = zip(*list(sorted_res))

    dot_similarity_list = list(dot_similarity)
    document_list = list(document)

    print("=========Top 10 most similar documents=========")
    print("Document", "Dot-Similarity value")
    for doc, sim in zip(document_list[:10], dot_similarity_list[:10]):
        print(doc, sim[0])


def main():
    vocabulary = get_unique_words()
    document_term_matrix = get_document_term_matrix(vocabulary)

    # execute for each query
    for filename in os.listdir(QUERY_PATH):
        print()
        print("RESULT for Query:{}".format(filename))
        query_term_matrix = get_query_term_matrix(filename, vocabulary)
        compute_dot_product_similarity(document_term_matrix.values, query_term_matrix.values)


if __name__ == '__main__':
    main()

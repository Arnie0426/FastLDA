from collections import defaultdict
from fastlda import LDA


def load_data(path="./data/nips.txt", min_term_occ=5):
    docs, vocabulary = [], []
    term_occurrences = defaultdict(int)
    try:
        for line in open(path).readlines():
            doc = line.split()
            docs.append(doc)
            for term in doc:
                term_occurrences[term] += 1
        vocabulary = [term for term, occ in term_occurrences.items()
                      if occ >= min_term_occ]
        inv_voc = {term: t for t, term in enumerate(vocabulary)}

        for d, doc in enumerate(docs):
            docs[d] = [inv_voc[t] for t in doc if t in inv_voc]
    except Exception as e:
        print("Failed to load dataset with exception: ", e)
    return docs, vocabulary


def train_lda(docs, V, K=50, alpha=1.0, beta=0.01):
    lda = LDA(docs, V, K, alpha, beta)
    lda.estimate(100, True)
    return {'topic_term_matrix': lda.getTopicTermMatrix(),
            'doc_topic_matrix': lda.getDocTopicMatrix()}


def show_topic_terms(TTM, vocabulary, num_terms=10):
    for t in range(len(TTM)):
        s = sorted(range(len(TTM[t])), key=TTM[t].__getitem__, reverse=True)
        print("\nTopic {}\n".format(t))
        print([vocabulary[w] for w in s[:num_terms]])


if __name__ == "__main__":
    docs, vocabulary = load_data()
    params = train_lda(docs, len(vocabulary))
    show_topic_terms(params['topic_term_matrix'], vocabulary)

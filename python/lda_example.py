from collections import defaultdict
from lda import LatentDirichletAllocation


def load_data(path="./data/nips.txt", min_term_occ=5):
    docs, vocabulary = [], []
    term_occurrences = defaultdict(int)
    try:
        for line in open(path).readlines():
            doc = line.split(', ')
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


def main():
    print("Loading NIPS dataset.")
    docs, vocabulary = load_data()
    print("NIPS dataset loaded.")
    # test training
    lda = LatentDirichletAllocation()
    print("Initialized LDA. Estimating parameters with 50 iterations.")
    lda.train(docs=docs, vocabulary=vocabulary, num_iterations=50)
    lda.save_parameters()
    print("Saved LDA Parameters.")
    # test inference model after training
    topics = lda.infer_doc(docs[0])
    s = sorted(range(len(topics)), key=topics.__getitem__, reverse=True)
    print(["(Top {} : {})".format(w, topics[w]) for w in s[:10]])
    print("\n\nComparing to doc in training set..")
    topics = lda.doc_topic_matrix[0]
    s = sorted(range(len(topics)), key=topics.__getitem__, reverse=True)
    print(["(Top {} : {})".format(w, topics[w]) for w in s[:10]])
    # Save topic terms to file
    lda.save_topic_terms()

    # Load model from scratch from file
    lda = LatentDirichletAllocation(model_path="./data/ttm.mat")
    topics = lda.infer_doc(docs[0])
    print("\n\nInferred model:")
    s = sorted(range(len(topics)), key=topics.__getitem__, reverse=True)
    print(["(Top {} : {})".format(w, topics[w]) for w in s[:10]])


if __name__ == "__main__":
    main()

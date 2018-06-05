import fastlda


def _load_mat_from_file(path, delimitter="\t"):
    with open(path, "r") as model_file:
        content = model_file.readlines()
    return [list(map(float, line.rstrip().split(delimitter)))
            for line in content]


class LatentDirichletAllocation(object):
    '''Python wrapper around the C++ modules'''

    def __init__(self, model_path=None, num_topics=None, 
                 alpha=None, beta=0.01):
        if not model_path and num_topics is None:
            raise LDAException("Please set the number of topics")
        self.topic_term_matrix = [] if \
            not model_path else _load_mat_from_file(model_path)
        self.num_topics = num_topics if num_topics else len(self.topic_term_matrix)
        # use suggested heuristic for alpha
        self.alpha = alpha if alpha else 50. / self.num_topics
        self.beta = beta
        self.inf_model = None if not self.topic_term_matrix else \
            fastlda.LDAInference(self.topic_term_matrix, self.alpha)

    def train(self, docs, vocabulary, num_topics=20,
              num_iterations=100, calculate_perplexity=False):
        if not docs or not vocabulary:
            raise LDAException("docs and vocabulary cannot be empty "
                               "during training")

        self.num_topics, self.vocabulary = num_topics, vocabulary
        lda = fastlda.LightLDA(docs, len(vocabulary), num_topics, self.alpha, self.beta)
        lda.estimate(num_iterations, calc_perp=calculate_perplexity)
        self.doc_topic_matrix = lda.getDocTopicMatrix()
        self.topic_term_matrix = lda.getTopicTermMatrix()

    def save_parameters(self, dtm_path="./data/dtm.mat",
                        ttm_path="./data/ttm.mat"):
        if not self.doc_topic_matrix or not self.topic_term_matrix:
            raise LDAException("LDA matrices are not trained.")
        try:
            with open(ttm_path, "w") as ttm_file:
                for row in self.topic_term_matrix:
                    ttm_file.write("\t".join(map(str, row)) + "\n")
            with open(dtm_path, "w") as dtm_file:
                for row in self.doc_topic_matrix:
                    dtm_file.write("\t".join(map(str, row)) + "\n")
        except Exception as e:
            print("Couldn't save matrices. Exception: ", e)

    def save_topic_terms(self, path="./data/topic_terms.txt", num_terms=10):
        with open(path, "w") as topic_term_file:
            for t, row in enumerate(self.topic_term_matrix):
                s = sorted(range(len(row)), key=row.__getitem__, reverse=True)
                topic_term_file.write("Topic {}: ".format(t))
                topic_term_file.write(", ".join([self.vocabulary[w]
                                                 for w in s[:num_terms]]))
                topic_term_file.write("\n\n")

    def infer_doc(self, doc, num_iterations=50):
        if not self.topic_term_matrix:
            raise LDAException("Topic Term Matrix needs to be loaded for "
                               "document inference")

        if not self.inf_model:
            self.inf_model = fastlda.LDAInference(self.topic_term_matrix, self.alpha)
        return self.inf_model.infer(doc, num_iterations)


class LDAException(Exception):
    pass

// Copyright 2018 Arnab Bhadury
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "cpp/lda.h"

namespace fastlda {
LDA::LDA(const vector<vector<size_t>> &docs, const size_t V,
         const size_t K, const float alpha, const float beta) : docs(docs),
             K(K), alpha(alpha), beta(beta), V(V),
             Z(vector<vector<size_t>> (docs.size())),
             CDK(vector<vector<size_t>> (docs.size(), std::vector<size_t> (K))),
             CKW(vector<vector<size_t>> (K, std::vector<size_t> (V))),
             CK(vector<size_t> (K)), total_num_of_words(0) {
    initialize();
}

void LDA::initialize() {
    uniform_int_distribution<> uniform_topic_dist(0, K-1);
    for (size_t d = 0; d < docs.size(); ++d) {
        size_t N = docs[d].size();
        Z[d] = vector<size_t>(N);
        total_num_of_words += N;

        for (size_t n = 0; n < N; ++n) {
            size_t term_id = docs[d][n];
            size_t topic_id = uniform_topic_dist(generator);
            CDK[d][topic_id]++;
            CKW[topic_id][term_id]++;
            CK[topic_id]++;
            Z[d][n] = topic_id;
        }
    }
}

void LDA::estimate(size_t num_iterations, bool calc_perp) {
    vector<float> prob_vector(K);
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        cout << "Iteration " << iter << endl;
        for (size_t d = 0; d < docs.size(); ++d) {
            size_t N = docs[d].size();
            for (size_t n = 0; n < N; ++n) {
                size_t topic_id = Z[d][n];
                size_t term_id = docs[d][n];

                // ignore current count
                CDK[d][topic_id]--;
                CKW[topic_id][term_id]--;
                CK[topic_id]--;

                for (size_t k = 0; k < K; ++k) {
                    prob_vector[k] = ((CDK[d][k] + alpha) *
                        (CKW[k][term_id] + beta)) / (CK[k] + V * beta);
                }

                discrete_distribution<size_t> mult(prob_vector.begin(),
                                                   prob_vector.end());

                topic_id = mult(generator);
                Z[d][n] = topic_id;
                CDK[d][topic_id]++;
                CKW[topic_id][term_id]++;
                CK[topic_id]++;
            }
        }
        if (calc_perp) {
            if (iter && (iter % 10 == 0 || iter == num_iterations - 1)) {
                cout << "Perplexity: " << calculate_perplexity() << endl;
            }
        }
    }
}

float LDA::calculate_perplexity() {
    float log_likelihood = 0;
    for (size_t d = 0; d < docs.size(); ++d) {
        for (size_t n = 0; n < docs[d].size(); ++n) {
            float likelihood = 0;
            size_t term_id = docs[d][n];
            size_t N = docs[d].size();
            for (size_t k = 0; k < K; ++k) {
                likelihood += ((CDK[d][k] + alpha) * (CKW[k][term_id] + beta))
                    / ((K * alpha + N) * (CK[k] + V * beta));
            }
            log_likelihood += log(likelihood);
        }
    }
    return exp(-log_likelihood / total_num_of_words);
}

vector<vector<float>> LDA::getDocTopicMatrix() const {
    vector<vector<float>> theta(docs.size(), vector<float> (K));
    for (size_t d = 0; d < docs.size(); ++d) {
        size_t N = docs[d].size();
        for (size_t k = 0; k < K; ++k) {
            theta[d][k] = (CDK[d][k] + alpha) / (N + K * alpha);
        }
    }
    return theta;
}

vector<vector<float>> LDA::getTopicTermMatrix() const {
    vector<vector<float>> phi(K, vector<float>(V));
    for (size_t k = 0; k < K; ++k) {
        for (size_t v = 0; v < V; ++v) {
            phi[k][v] = (CKW[k][v] + beta) / (CK[k] + V * beta);
        }
    }
    return phi;
}
}

// Copyright 2018 Arnab Bhadury
#include <algorithm>
#include <random>
#include <vector>

#include "cpp/lda_inference.h"

namespace fastlda {
LDAInference::LDAInference(const vector<vector<float>> &topic_term_matrix,
                           const float alpha) :
                               topic_term_matrix(topic_term_matrix),
                               alpha(alpha),
                               K(topic_term_matrix.size()) { }

vector<float> LDAInference::infer(const vector<size_t> &doc,
                                  size_t num_iterations) const {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine local_generator(rd());
#else
    static random_device rd;
    static default_random_engine local_generator(rd());
#endif

    size_t N = doc.size();
    vector<size_t> cdk(K);
    vector<float> prob_vector(K);
    vector<size_t> topic_indices(N);

    // initialize counts
    uniform_int_distribution<> uniform_topic_dist(0, K-1);
    for (size_t n = 0; n < N; ++n) {
        size_t term_id = doc[n];
        size_t topic_id = uniform_topic_dist(local_generator);
        cdk[topic_id]++;
        topic_indices[n] = topic_id;
    }

    // infer topics for document
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t n = 0; n < N; ++n) {
            size_t topic_id = topic_indices[n];
            size_t term_id = doc[n];
            cdk[topic_id]--;

            for (size_t k = 0; k < K; ++k) {
                prob_vector[k] = (cdk[k] + alpha)
                                 * topic_term_matrix[k][term_id];
            }
            discrete_distribution<size_t> mult(prob_vector.begin(),
                                               prob_vector.end());
            topic_id = mult(local_generator);
            topic_indices[n] = topic_id;
            cdk[topic_id]++;
        }
    }
    // compute true probability vector
    float sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        prob_vector[k] = cdk[k] + alpha;
        sum += prob_vector[k];
    }
    for_each(prob_vector.begin(), prob_vector.end(),
            [&sum](float &p) { p /= sum; });
    return prob_vector;
}
}  // namespace fastlda

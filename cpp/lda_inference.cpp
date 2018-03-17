#include <algorithm>
#include <random>
#include <vector>

#include "lda_inference.h"

LDA_Inference::LDA_Inference(const vector<vector<float>> &topic_term_matrix,
                             const float alpha) :
                             // again note that as your code stands, you are copying data here, probably needlessly
                                 topic_term_matrix(topic_term_matrix),
                                 alpha(alpha),
                                 K(topic_term_matrix.size()) {

}

vector<float> LDA_Inference::infer(const vector<size_t> &doc,
                                   size_t num_iterations) const {

#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine local_generator(rd());
#else
    static random_device rd;
    static default_random_engine local_generator(rd());
#endif

    auto N = doc.size();
    auto cdk = vector<size_t>(K);
    auto prob_vector = vector<float>(K);
    auto topic_indices = vector<size_t>(N);

    // initialize counts
    auto uniform_topic_dist = uniform_int_distribution<>(0, K-1);
    // std::transform populating topic_indices and the lamdba to transform taking cdk by reference so it can update that too
    for (auto n = 0; n < N; ++n) {
        auto term_id = doc[n];
        auto topic_id = uniform_topic_dist(local_generator);
        cdk[topic_id]++;
        topic_indices[n] = topic_id;
    }

    // infer topics for document
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // you can probably use a HOF here too, see http://en.cppreference.com/w/cpp/algorithm 
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
    // std::accumulate is your friend
    for (size_t k = 0; k < K; ++k) {
        prob_vector[k] = cdk[k] + alpha;
        sum += prob_vector[k];
    }
    // std::accumulate here too
    for_each(prob_vector.begin(), prob_vector.end(),
            [&sum](float &p) { p /= sum; });
    return prob_vector;
}

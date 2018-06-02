// Copyright 2018 Arnab Bhadury
#include <vector>
#include <random>

#include "cpp/cgs_lda.h"

namespace fastlda {
CGS_LDA::CGS_LDA(const vector<vector<size_t>> &docs, const size_t V,
         const size_t K, const float alpha, const float beta)
            : LDA(docs, V, K, alpha, beta) {
}

void CGS_LDA::estimate(size_t num_iterations, bool calc_perp) {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine generator(rd());
#else
    static random_device rd;
    static default_random_engine generator(rd());
#endif
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
}  // namespace fastlda

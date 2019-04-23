// Copyright 2018 Arnab Bhadury
#include <vector>
#include <random>

#include "cpp/cgs_lda.h"

namespace fastlda {
CGS_LDA::CGS_LDA(const vector<vector<size_t>> &docs, const size_t V,
         const size_t K, const float alpha, const float beta)
            : LDA(docs, V, K, alpha, beta) {
}

void CGS_LDA::estimate(size_t numIterations, bool calcPerp) {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine generator(rd());
#else
    static random_device rd;
    static default_random_engine generator(rd());
#endif
    vector<float> probVector(numTopics_);
    for (auto iter = 0; iter < numIterations; ++iter) {
        cout << "Iteration " << iter << endl;
        for (auto d = 0; d < docs_.size(); ++d) {
            auto N = docs_[d].size();
            for (auto n = 0; n < N; ++n) {
                auto topicId = Z[d][n];
                auto termId = docs_[d][n];

                // ignore current count
                CDK.coeffRef(d, topicId)--;
                CKW.coeffRef(topicId, termId)--;
                CK[topicId]--;

                for (auto k = 0; k < numTopics_; ++k) {
                    probVector[k] = ((CDK.coeff(d, k) + alpha_) *
                        (CKW.coeff(k, termId) + beta_)) / (CK[k] + vocSize_ * beta_);
                }

                discrete_distribution<size_t> mult(probVector.begin(),
                                                   probVector.end());

                topicId = mult(generator);
                Z[d][n] = topicId;
                CDK.coeffRef(d, topicId)++;
                CKW.coeffRef(topicId, termId)++;
                CK[topicId]++;
            }
        }
        if (calcPerp) {
            if (iter && (iter % 10 == 0 || iter == numIterations - 1)) {
                cout << "Perplexity: " << calculatePerplexity() << endl;
            }
        }
    }
}
}  // namespace fastlda

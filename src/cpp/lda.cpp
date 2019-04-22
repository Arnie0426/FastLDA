// Copyright 2018 Arnab Bhadury
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "cpp/lda.h"

namespace fastlda {
LDA::LDA(const vector<vector<size_t>> &docs, const size_t V,
         const size_t K, const float alpha, const float beta) : docs_(docs),
             numTopics_(K), alpha_(alpha), beta_(beta), vocSize_(V),
             Z(build2DVector<size_t>(docs.size(), 0)),
             CDK(build2DVector<size_t>(docs.size(), K)),
             CKW(build2DVector<size_t>(K, V)),
             CK(vector<size_t> (K)), totalNumOfWords_(0) {
    initialize();
}

void LDA::initialize() {
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<> uniformTopicDist(0, numTopics_-1);
    for (auto d = 0; d < docs_.size(); ++d) {
        auto N = docs_[d].size();
        Z[d] = vector<size_t>(N);
        totalNumOfWords_ += N;

        for (auto n = 0; n < N; ++n) {
            auto termId = docs_[d][n];
            auto topicId = uniformTopicDist(generator);
            CDK[d][topicId]++;
            CKW[topicId][termId]++;
            CK[topicId]++;
            Z[d][n] = topicId;
        }
    }
}

float LDA::calculatePerplexity() {
    float logLikelihood = 0;
    for (auto d = 0; d < docs_.size(); ++d) {
        for (auto n = 0; n < docs_[d].size(); ++n) {
            float likelihood = 0;
            auto termId = docs_[d][n];
            auto N = docs_[d].size();
            for (auto k = 0; k < numTopics_; ++k) {
                likelihood += ((CDK[d][k] + alpha_) * (CKW[k][termId] + beta_))
                    / ((numTopics_ * alpha_ + N) * (CK[k] + vocSize_ * beta_));
            }
            logLikelihood += log(likelihood);
        }
    }
    return exp(-logLikelihood / totalNumOfWords_);
}

vector<vector<float>> LDA::getDocTopicMatrix() const {
    vector<vector<float>> theta(docs_.size(), vector<float> (numTopics_));
    for (auto d = 0; d < docs_.size(); ++d) {
        auto N = docs_[d].size();
        for (auto k = 0; k < numTopics_; ++k) {
            theta[d][k] = (CDK[d][k] + alpha_) / (N + numTopics_ * alpha_);
        }
    }
    return theta;
}

vector<vector<float>> LDA::getTopicTermMatrix() const {
    vector<vector<float>> phi(numTopics_, vector<float>(vocSize_));
    for (auto k = 0; k < numTopics_; ++k) {
        for (auto v = 0; v < vocSize_; ++v) {
            phi[k][v] = (CKW[k][v] + beta_) / (CK[k] + vocSize_ * beta_);
        }
    }
    return phi;
}

vector<vector<size_t>> LDA::getSparseDocTopicMatrix() const {
    return CDK;
}

vector<vector<size_t>> LDA::getSparseTopicTermMatrix() const {
    return CKW;
}

vector<size_t> LDA::getTopicCountMatrix() const {
    return CK;
}
}  // namespace fastlda

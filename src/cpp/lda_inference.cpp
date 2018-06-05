// Copyright 2018 Arnab Bhadury
#include <algorithm>
#include <random>
#include <vector>

#include "cpp/lda_inference.h"

namespace fastlda {
LDAInference::LDAInference(const vector<vector<float>> &topicTermMatrix,
                           const float alpha) :
                               topicTermMatrix(topicTermMatrix),
                               alpha_(alpha),
                               numTopics_(topicTermMatrix.size()) { }

vector<float> LDAInference::infer(const vector<size_t> &doc,
                                  size_t numIterations) const {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine localGenerator(rd());
#else
    static random_device rd;
    static default_random_engine localGenerator(rd());
#endif

    auto N = doc.size();
    vector<size_t> cdk(numTopics_);
    vector<size_t> topicIndices(N);

    // initialize counts
    uniform_int_distribution<> uniformTopicDist(0, numTopics_-1);
    for (auto n = 0; n < N; ++n) {
        auto termId = doc[n];
        auto topicId = uniformTopicDist(localGenerator);
        cdk[topicId]++;
        topicIndices[n] = topicId;
    }

    {
        vector<float> probVector(numTopics_);
        // infer topics for document
        for (auto iter = 0; iter < numIterations; ++iter) {
            for (auto n = 0; n < N; ++n) {
                auto topicId = topicIndices[n];
                auto termId = doc[n];
                cdk[topicId]--;

                for (auto k = 0; k < numTopics_; ++k) {
                    probVector[k] = (cdk[k] + alpha_)
                                     * topicTermMatrix[k][termId];
                }
                discrete_distribution<size_t> mult(probVector.begin(),
                                                   probVector.end());
                topicId = mult(localGenerator);
                topicIndices[n] = topicId;
                cdk[topicId]++;
            }
        }
    }
    // compute true probability vector
    vector<float> probVector(cdk.begin(), cdk.end());
    for_each(probVector.begin(), probVector.end(),
             [&](float &p) { p = (p +alpha_) / (N + numTopics_ * alpha_); });
    return probVector;
}
}  // namespace fastlda

// Copyright 2018 Arnab Bhadury
#ifndef CPP_LDA_H_
#define CPP_LDA_H_
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cpp/common.h"
namespace fastlda {
class LDA {
 protected:
    const vector<vector<size_t>> docs_;
    const size_t numTopics_;
    const size_t vocSize_;
    const float alpha_;
    const float beta_;
    size_t totalNumOfWords_;
    // Count matrices
    vector<vector<size_t>> Z;
    vector<vector<size_t>> CDK;
    vector<vector<size_t>> CKW;
    vector<size_t> CK;

 public:
    LDA(const vector<vector<size_t>> &docs, const size_t V, 
        const size_t K, const float alpha, const float beta);
    void initialize();
    virtual void estimate(size_t numIterations, bool calcPerp) { }
    float calculatePerplexity();
    vector<vector<float>> getTopicTermMatrix() const;
    vector<vector<float>> getDocTopicMatrix() const;
};
}  // namespace fastlda
#endif  // CPP_LDA_H_

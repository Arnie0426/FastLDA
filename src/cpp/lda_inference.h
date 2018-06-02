// Copyright 2018 Arnab Bhadury
#ifndef SRC_CPP_LDA_INFERENCE_H_
#define SRC_CPP_LDA_INFERENCE_H_

#include <vector>
#include <random>

#include "cpp/common.h"

namespace fastlda {
class LDAInference {
 private:
    const vector<vector<float>> topicTermMatrix;
    const size_t numTopics_;
    const float alpha_;

 public:
    LDAInference(const vector<vector<float>> &topicTermMatrix,
                 const float alpha);
    vector<float> infer(const vector<size_t> &doc,
                        size_t numIterations) const;
};
}  // namespace fastlda
#endif  // SRC_CPP_LDA_INFERENCE_H_

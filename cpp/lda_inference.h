// Copyright 2018 Arnab Bhadury
#ifndef CPP_LDA_INFERENCE_H_
#define CPP_LDA_INFERENCE_H_

#include <vector>
#include <random>

#include "cpp/common.h"

namespace fastlda {
class LDA_Inference {
 private:
    vector<vector<float>> topic_term_matrix;
    size_t K;
    float alpha;

 public:
    LDA_Inference(const vector<vector<float>> &topic_term_matrix,
                  const float alpha);
    vector<float> infer(const vector<size_t> &doc,
                        size_t num_iterations) const;
};
}  // namespace fastlda
#endif  // CPP_LDA_INFERENCE_H_

// Copyright 2018 Arnab Bhadury
#ifndef SRC_CPP_CGS_LDA_H_
#define SRC_CPP_CGS_LDA_H_

#include <vector>

#include "cpp/lda.h"
namespace fastlda {
class CGS_LDA : public LDA {
 public:
    CGS_LDA(const vector<vector<size_t>> &docs, const size_t V,
           const size_t K, const float alpha, const float beta);
    void estimate(size_t numIterations, bool calcPerp);
};
}  // namespae fastlda
#endif  // SRC_CPP_CGS_LDA_H_

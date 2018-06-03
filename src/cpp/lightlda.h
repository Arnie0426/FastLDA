// Copyright 2018 Arnab Bhadury
#ifndef SRC_CPP_LIGHTLDA_H_
#define SRC_CPP_LIGHTLDA_H_

#include <vector>

#include "cpp/lda.h"
#include "cpp/alias_table.h"

namespace fastlda {
class LightLDA : public LDA {
 public:
    LightLDA(const vector<vector<size_t>> &docs, const size_t V,
             const size_t K, const float alpha, const float beta);
    void estimate(size_t numIterations, size_t numMHSteps, bool calcPerp);

 private:
    void buildAliasTables();
    void buildBetaAliasTable();
    void buildTermAliasTable(size_t w);

    vector<size_t> betaSamples;
    vector<vector<size_t>> termSamples;
    vector<size_t> sampleCounts;

    float betaSum;
    float termSum;
    uniform_int_distribution<> uniformTopic;
    uniform_real_distribution<float> u01;
};
}  // namespace fastlda
#endif  // SRC_CPP_LIGHTLDA_H_

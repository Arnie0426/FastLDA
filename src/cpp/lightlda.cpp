// Copyright 2018 Arnab Bhadury
#include <vector>

#include "cpp/lightlda.h"

namespace fastlda {
LightLDA::LightLDA(const vector<vector<size_t>> &docs, const size_t V,
                   const size_t K, const float alpha, const float beta)
                    : LDA(docs, V, K, alpha, beta),
                      betaSamples(vector<size_t>(K)),
                      termSamples(vector<vector<size_t>>(V,
                        vector<size_t>(K))),
                      sampleCounts(vector<size_t>(V + 1)),
                      uniformTopic(uniform_int_distribution<>(0, K - 1)),
                      u01(uniform_real_distribution<float>(0, 1)),
                      betaSum(0.0), termSum(0.0) {
    buildAliasTables();
}

void LightLDA::buildAliasTables() {
    buildBetaAliasTable();
    for (auto w = 0; w < vocSize_; ++w) {
        buildTermAliasTable(w);
        for (auto k = 0; k < numTopics_; ++k) {
            termSum += CKW[k][w] / (CK[k] + vocSize_ * beta_);
        }
    }
    for (size_t k = 0; k < numTopics_; ++k) {
        betaSum += (beta_ / CK[k] + vocSize_ * beta_);
    }
}
void LightLDA::buildBetaAliasTable() {
    vector<float> p(numTopics_);
    for (auto k = 0; k < numTopics_; ++k) {
        p[k] = beta_ / (CK[k] + vocSize_ * beta_);
    }
    AliasTable betaAliasTable(p);
    for (auto k = 0; k < numTopics_; ++k) {
        betaSamples[k] = betaAliasTable.getAliasSample();
    }
}

void LightLDA::buildTermAliasTable(size_t w) {
    vector<float> p(numTopics_);
    for (auto k = 0; k < numTopics_; ++k) {
        p[k] = CKW[k][w] / (CK[k] + vocSize_ * beta_);
    }
    AliasTable termAliasTable(p);
    for (auto k = 0; k < numTopics_; ++k) {
        termSamples[w][k] = termAliasTable.getAliasSample();
    }
}

void LightLDA::estimate(size_t numIterations, size_t numMHSteps,
                        bool calcPerp) {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine generator(rd());
#else
    static random_device rd;
    static default_random_engine generator(rd());
#endif
    for (auto iter = 0; iter < numIterations; ++iter) {
        cout << "Iteration " << iter << endl;
        size_t sampleCount = 0;
        size_t acceptCount = 0;
        for (auto d = 0; d < docs_.size(); ++d) {
            auto N = docs_[d].size();
            uniform_int_distribution<> docDist(0, N - 1);
            for (auto n = 0; n < N; ++n) {
                for (auto m = 0; m < numMHSteps; ++m) {
                    sampleCount++;

                    size_t topicId = Z[d][n];
                    size_t termId = docs_[d][n];

                    size_t proposedTopic;
                    float accept = 0.0;

                    // ignore current count
                    CDK[d][topicId]--;
                    CKW[topicId][termId]--;
                    CK[topicId]--;

                    uniform_int_distribution<> coin(0, 1);  // coin flip
                    auto proposal = coin(generator);
                    if (proposal == 0) {
                        float u = u01(generator) * (N - 1 + numTopics_ * alpha_);
                        if (u < N - 1) {
                            size_t index = docDist(generator);
                            proposedTopic = Z[d][index];
                        } else {
                            proposedTopic = uniformTopic(generator);
                        }
                        // MH Probability
                        accept = (CKW[proposedTopic][termId] + beta_)
                                 / (CKW[topicId][termId] + beta_);
                        accept *= (CK[topicId] + vocSize_ * beta_)
                                  / (CK[proposedTopic] + vocSize_ * beta_);
                        accept *= (CDK[d][proposedTopic] + alpha_)
                                  / (CDK[d][topicId] + alpha_);
                        accept *= (CDK[d][topicId] + alpha_ + 1)
                                  / (CDK[d][proposedTopic] + alpha_ + 1);
                    } else {
                        if (sampleCounts[vocSize_] >= numTopics_) {
                            buildBetaAliasTable();
                            sampleCounts[vocSize_] = 0;
                        }
                        if (sampleCounts[termId] >= vocSize_) {
                            buildTermAliasTable(termId);
                            sampleCounts[termId] = 0;
                        }
                        float u = u01(generator) * (betaSum + termSum);
                        if (u < betaSum) {
                            proposedTopic = betaSamples[sampleCounts[vocSize_]];
                            sampleCounts[vocSize_]++;
                        } else {
                            proposedTopic = termSamples[termId][sampleCounts[termId]];
                            sampleCounts[termId]++;
                        }
                        accept = (CDK[d][proposedTopic] + alpha_)
                                 / (CDK[d][topicId] + alpha_);
                        accept *= (CKW[proposedTopic][termId] + beta_)
                                  / (CKW[topicId][termId] + beta_);
                        accept *= (CK[topicId] + vocSize_ * beta_)
                                  / (CK[proposedTopic] + vocSize_ * beta_);
                        accept *= (CKW[topicId][termId] + beta_ + 1)
                                  / (CKW[proposedTopic][termId] + beta_ + 1);
                        accept *= (CK[proposedTopic] + 1 + vocSize_ * beta_)
                                  / (CK[topicId] + 1 + vocSize_ * beta_);
                    }
                    if (u01(generator) >= accept) {
                        // Reject proposed sample
                        proposedTopic = topicId;
                    } else {
                        acceptCount++;
                        betaSum -= (beta_ / (CK[topicId] + vocSize_ * beta_));
                        betaSum += (beta_ / (CK[proposedTopic] + vocSize_ * beta_));
                        termSum -= (CKW[topicId][termId]
                                   / (CK[topicId] + vocSize_ * beta_));
                        termSum += (CKW[proposedTopic][termId]
                                   / (CK[proposedTopic] + vocSize_ * beta_));
                    }
                    CDK[d][proposedTopic]++;
                    CKW[proposedTopic][termId]++;
                    CK[proposedTopic]++;
                    Z[d][n] = proposedTopic;
                }
            }
        }
        cout << "MH Acceptance: " << float(acceptCount) / sampleCount << endl;
        if (calcPerp) {
            if (iter && (iter % 10 == 0 || iter == numIterations - 1)) {
                cout << "Perplexity: " << calculatePerplexity() << endl;
            }
        }
    }
}
}  // namespace fastlda

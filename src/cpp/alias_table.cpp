// Copyright 2018 Arnab Bhadury
#include <limits>
#include <random>
#include <numeric>
#include <string>
#include <vector>
#include <utility>

#include "cpp/alias_table.h"

namespace fastlda {
AliasTable::AliasTable(const vector<float> &prob) {
    auto dim = prob.size();
    float norm = accumulate(prob.begin(), prob.end(), 0.0);
    vector<float> scaledProb(dim);

    vector<pair<float, size_t>> rich;
    vector<pair<float, size_t>> poor;

    // Scale probabilities, and divide into rich and poor lists
    for (auto i = 0; i < dim; ++i) {
        scaledProb[i] = dim * prob[i] / norm;
        if (scaledProb[i] < 1) {
            poor.push_back(make_pair(scaledProb[i], i));
        } else {
            rich.push_back(make_pair(scaledProb[i], i));
        }
    }

    while (!rich.empty() && !poor.empty()) {
        auto r = rich.back();
        auto p = poor.back();
        rich.pop_back();
        poor.pop_back();

        mainTable_.push_back(p);
        aliasTable_.push_back(r.second);

        auto remWeight = (p.first + r.first) - 1.0;
        if (remWeight < 1) {
            poor.push_back(make_pair(remWeight, r.second));
        } else {
            rich.push_back(make_pair(remWeight, r.second));
        }
    }

    // handle numerical instability
    while (!rich.empty()) {
        auto r = rich.back();
        rich.pop_back();
        mainTable_.push_back(make_pair(1.0, r.second));
        aliasTable_.push_back(-1);
    }

    while (!poor.empty()) {
        auto p = poor.back();
        poor.pop_back();
        mainTable_.push_back(make_pair(1.0, p.second));
        aliasTable_.push_back(-1);
    }
}

size_t AliasTable::getAliasSample() const {
#ifndef __APPLE__
    static thread_local random_device rd;
    static thread_local default_random_engine generator(rd());
#else
    static random_device rd;
    static default_random_engine generator(rd());
#endif

    if (mainTable_.empty()) {
        // Error
        return std::numeric_limits<size_t>::max();
    }
    auto dim = mainTable_.size();
    uniform_real_distribution<float> u01(0, 1);
    uniform_int_distribution<> uniformTable(0, dim - 1);

    // roll dice
    auto k = uniformTable(generator);
    if (aliasTable_[k] == -1) {
        return mainTable_[k].second;
    }

    auto cell = mainTable_[k];

    // flip coin
    auto coin = u01(generator);
    if (coin <= cell.first) {
        return cell.second;
    } else {
        return aliasTable_[k];
    }
}
}  // namespace fastlda

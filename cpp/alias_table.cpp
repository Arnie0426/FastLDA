// Copyright 2018 Arnab Bhadury
#include <random>
#include <numeric>
#include <string>
#include <vector>
#include <utility>

#include "alias_table.h"

namespace fastlda {
AliasTable::AliasTable(const vector<float> &prob) {
    size_t dim = prob.size();
    float norm = std::accumulate(prob.begin(), prob.end(), 0.0);
    vector<float> scaled_prob(dim);

    vector<pair<float, size_t>> rich;
    vector<pair<float, size_t>> poor;

    // Scale probabilities, and divide into rich and poor lists
    for (size_t i = 0; i < dim; ++i) {
        scaled_prob[i] = dim * prob[i] / norm;
        if (scaled_prob[i] < 1) {
            poor.push_back(make_pair(scaled_prob[i], i));
        } else {
            rich.push_back(make_pair(scaled_prob[i], i));
        }
    }

    while (!rich.empty() && !poor.empty()) {
        pair<float, size_t> r = rich.back();
        pair<float, size_t> p = poor.back();
        rich.pop_back();
        poor.pop_back();

        main_table.push_back(p);
        alias_table.push_back(r.second);

        float rem_weight = (p.first + r.first) - 1.0;
        if (rem_weight < 1) {
            poor.push_back(make_pair(rem_weight, r.second));
        } else {
            rich.push_back(make_pair(rem_weight, r.second));
        }
    }

    // handle numerical instability
    while (!rich.empty()) {
        pair<float, size_t> r = rich.back();
        rich.pop_back();
        main_table.push_back(make_pair(1.0, r.second));
        alias_table.push_back(-1);
    }

    while (!poor.empty()) {
        pair<float, size_t> p = poor.back();
        poor.pop_back();
        main_table.push_back(make_pair(1.0, p.second));
        alias_table.push_back(-1);
    }

}

size_t AliasTable::get_alias_sample(default_random_engine generator) const {
    if (main_table.empty()) {
        // Error
        return std::numeric_limits<size_t>::max();
    }
    size_t dim = main_table.size();
    uniform_real_distribution<float> u01(0, 1);
    uniform_int_distribution<> uniform_table(0, dim - 1);

    // roll dice
    size_t k = uniform_table(generator);
    if (alias_table[k] == -1) {
        return main_table[k].second;
    }

    pair<float, size_t> cell = main_table[k];

    // flip coin
    float coin = u01(generator);
    if (coin <= cell.first) {
        return cell.second;
    } else {
        return alias_table[k];
    }
}
}  // fastlda

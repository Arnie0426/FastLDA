#include <random>
#include <numeric>
#include <string>
#include <vector>
#include <utility>

#include "alias_table.cpp"

AliasTable::AliasTable(const vector<float> prob) {
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
        alias_table.push_back(r[1]);

        float rem_weight = (p[0] + r[0]) - 1.0;
        if (rem_weight < 1) {
            poor.push_back(make_pair(rem_weight, r[1]));
        } else {
            rich.push_back(make_pair(rem_weight, r[1]));
        }
    }

    // handle numerical instability
    while (!rich.empty()) {
        pair<float, size_t> r = rich.back();
        rich.pop_back();
        main_table.push_back(make_pair(1.0, r[1]));
        alias_table.push_back(-1);
    }

    while (!poor.empty()) {
        pair<float, size_t> p = poor.back();
        poor.pop_back();
        main_table.push_back(make_pair(1.0, p[1]));
        alias_table.push_back(-1);
    }

}

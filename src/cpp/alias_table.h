// Copyright 2018 Arnab Bhadury
#ifndef CPP_ALIAS_TABLE_H_
#define CPP_ALIAS_TABLE_H_

#include <algorithm>
#include <vector>
#include <random>
#include <utility>

#include "cpp/common.h"

namespace fastlda {
class AliasTable {
 public:
    explicit AliasTable(const vector<float> &prob);
    size_t get_alias_sample(default_random_engine) const;
 private:
    vector<pair<float, size_t>> main_table;
    vector<size_t> alias_table;
};
}  // namespace fastlda
#endif  // CPP_ALIAS_TABLE_H_

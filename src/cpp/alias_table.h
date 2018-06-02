// Copyright 2018 Arnab Bhadury
#ifndef SRC_CPP_ALIAS_TABLE_H_
#define SRC_CPP_ALIAS_TABLE_H_

#include <algorithm>
#include <vector>
#include <random>
#include <utility>

#include "cpp/common.h"

namespace fastlda {
class AliasTable {
 public:
    explicit AliasTable(const vector<float> &prob);
    size_t getAliasSample() const;
 private:
    vector<pair<float, size_t>> mainTable_;
    vector<size_t> aliasTable_;
};
}  // namespace fastlda
#endif  // SRC_CPP_ALIAS_TABLE_H_

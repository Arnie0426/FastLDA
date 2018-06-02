// Copyright 2018 Arnab Bhadury
#ifndef CPP_ALIAS_TABLE_H
#define CPP_ALIAS_TABLE_H

#include <algorithm>
#include <random>
#include <vector>
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
}  // fastlda
#endif  /* CPP_ALIAS_TABLE_H */
=======
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
using namespace std;

class AliasTable {
 public:
    explicit AliasTable(const vector<float> &prob);
    size_t get_alias_sample(default_random_engine) const;
 private:
    vector<pair<float, size_t>> main_table;
    vector<size_t> alias_table;
};

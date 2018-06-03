// Copyright 2018 Arnab Bhadury

#ifndef SRC_CPP_COMMON_H_
#define SRC_CPP_COMMON_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fastlda {
using std::string;
using std::vector;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::discrete_distribution;
using std::uniform_real_distribution;
using std::random_device;
using std::cout;
using std::endl;
using std::accumulate;
using std::pair;
using std::make_pair;

template<class T>
static vector<vector<T>> build2DVector(size_t x_dim, size_t y_dim) {
  return vector<vector<T>>(x_dim, vector<T>(y_dim));
}

}  // namespace fastlda
#endif  // SRC_CPP_COMMON_H_

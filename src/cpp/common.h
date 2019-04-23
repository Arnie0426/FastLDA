// Copyright 2018 Arnab Bhadury

#ifndef SRC_CPP_COMMON_H_
#define SRC_CPP_COMMON_H_

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
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

using std::chrono::time_point;
using std::chrono::steady_clock;

typedef Eigen::SparseMatrix<size_t> SparseCountMatrix;

template<class T>
static vector<vector<T>> build2DVector(size_t xDim, size_t yDim) {
  return vector<vector<T>>(xDim, vector<T>(yDim));
}

static SparseCountMatrix buildSparseCountMatrix(size_t xDim, size_t yDim) {
  return SparseCountMatrix(xDim, yDim);
}

}  // namespace fastlda
#endif  // SRC_CPP_COMMON_H_

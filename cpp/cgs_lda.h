// Copyright 2018 Arnab Bhadury
#ifndef CPP_CGS_LDA_H
#define CPP_CGS_LDA_H

#include "cpp/lda.h"

namespace fastlda {
  class CGS_LDA : public LDA {
   public:
      CGS_LDA(const vector<vector<size_t>> &docs, const size_t V,
             const size_t K, const float alpha, const float beta);
      void estimate(size_t num_iterations, bool calc_perp);
  };
}  // namespace fastlda
#endif  /* CPP_CGS_LDA_H */

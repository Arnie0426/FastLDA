#include "lda.h"

class CGS_LDA : public LDA {
 public:
    CGS_LDA(const vector<vector<size_t>> &docs, const size_t V,
           const size_t K, const float alpha, const float beta);
    void estimate(size_t num_iterations, bool calc_perp);
};
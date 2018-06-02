#include "lda.h"
#include "alias_table.h"

class LightLDA : public LDA {
 public:
    LightLDA(const vector<vector<size_t>> &docs, const size_t V, const size_t K,
             const float alpha, const float beta);
    void estimate(size_t num_iterations, size_t num_mh_steps, bool calc_perp);

 private:
    void build_beta_alias_table();
    void build_term_alias_table(size_t w);

    vector<size_t> beta_samples;
    vector<vector<size_t>> term_samples;
    vector<size_t> sample_counts;

    float beta_sum;
    float term_sum;
    uniform_int_distribution<> uniform_topic;
    uniform_real_distribution<float> u01;
};
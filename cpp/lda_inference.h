#include <vector>
#include <random>
using namespace std;

class LDA_Inference {
 private:
    vector<vector<float>> topic_term_matrix;
    size_t K;
    float alpha;

 public:
    LDA_Inference(const vector<vector<float>> &topic_term_matrix,
                  const float alpha);
    vector<float> infer(const vector<size_t> &doc, size_t num_iterations) const;
};

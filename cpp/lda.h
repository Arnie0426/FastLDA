#include <iostream>
#include <random>
#include <string>
#include <vector>
// using namespace in header files is poor practice, since it affects the use of namespaces in client code that #includes your header. 
// Doing this in cpp files is normal, since then it will only apply to the single file
using namespace std;

class LDA {
 private:
    // c++ convention uses _ prefix on member variables
    // Note that this vector is copied in the constructor. I think you want this to be a reference, just make sure it never ends up dangling.
    // everything that can be const should be
    const vector<vector<size_t>>& docs; // this should probaly be a const reference
    size_t K;
    size_t V;
    float alpha;
    float beta;
    size_t total_num_of_words;
    // Count matrices
    vector<vector<size_t>> Z;
    vector<vector<size_t>> CDK;
    vector<vector<size_t>> CKW;
    vector<size_t> CK;

    default_random_engine generator;

    template<class T>
    static vector<vector<T>> build2dvector(int x, int y)
    {
        return vector<vector<T>>(x, vector<T>(y));
    }

 public:
    LDA(const vector<vector<size_t>> &docs,
        const size_t V, const size_t K, const float alpha, const float beta);
    void initialize();
    void estimate(size_t num_iterations, bool calc_perp);
    float calculate_perplexity();
    vector<vector<float>> getTopicTermMatrix() const;
    vector<vector<float>> getDocTopicMatrix() const;
};

#include <vector>
#include <random>
#include <utility>
using namespace std;

class AliasTable {
 public:
    explicit AliasTable(const vector<float> &prob);
    size_t get_alias_sample(std::default_random_generator)const;
 private:
    vector<pair<float, size_t>> main_table;
    vector<size_t> alias_table;
};
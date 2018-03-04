#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "lda.h"

// if you could *move* the data in the vector, leaving an invalid vector at the call site, this is what it would look like
LDA::LDA(const vector<vector<size_t>> &&docs, const size_t V,
         const size_t K, const float alpha, const float beta) : docs(std::move(docs)),
            // identical parameter and member variable names make this look really funky, as a seasoned c++ dev I go "does that even work?"
            // _ prefix on member variable names fixes this. You can also use 'this.' prefix
             K(K), this.alpha(alpha), beta(beta), V(V),
             // can simplify these constructor calls quite a bit here, see (2) in http://en.cppreference.com/w/cpp/container/vector/vector.
             /// Also, since these are quite complex vectors, I would create some static private member functions and delegate. They could
             // also very naturally take on some of the work of initialize()
             Z(buildZvector(docs.size()),
             // picky: if you're "using namespace std", then you don't need the std prefixes. I actually prefer no "using" statement and 
             // prefixed, but do one or the other
             CDK(docs.size(), std::vector<size_t>(K)),
             CKW(K, std::vector<size_t>(V)),
             CK(vector<size_t> (K)), total_num_of_words(0) {
    initialize();
}

void LDA::initialize() {
    // use auto whenever possibe, which is most of the time
    auto uniform_topic_dist = uniform_int_distribution<>(0, K-1);
    for (auto d = 0; d < docs.size(); ++d) {
        auto N = docs[d].size();
        Z[d] = vector<size_t>(N);
        total_num_of_words += N;

        for (auto n = 0; n < N; ++n) {
            auto term_id = docs[d][n];
            auto topic_id = uniform_topic_dist(generator);
            CDK[d][topic_id]++;
            CKW[topic_id][term_id]++;
            CK[topic_id]++;
            Z[d][n] = topic_id;
        }
    }
}

void LDA::estimate(size_t num_iterations, bool calc_perp) {
    auto prob_vector = vector<float>(K);
    for (auto iter = 0; iter < num_iterations; ++iter) {
        cout << "Iteration " << iter << endl;
        for (auto d = 0; d < docs.size(); ++d) {
            auto N = docs[d].size();
            for (auto n = 0; n < N; ++n) {
                auto topic_id = Z[d][n];
                auto term_id = docs[d][n];

                // ignore current count
                CDK[d][topic_id]--;
                CKW[topic_id][term_id]--;
                CK[topic_id]--;

                for (size_t k = 0; k < K; ++k) {
                    prob_vector[k] = ((CDK[d][k] + alpha) *
                        (CKW[k][term_id] + beta)) / (CK[k] + V * beta);
                }

                discrete_distribution<size_t> mult(prob_vector.begin(),
                                                   prob_vector.end());

                topic_id = mult(generator);
                Z[d][n] = topic_id;
                CDK[d][topic_id]++;
                CKW[topic_id][term_id]++;
                CK[topic_id]++;
            }
        }
        if (calc_perp) {
            if (iter && (iter % 10 == 0 || iter == num_iterations - 1)) {
                cout << "Perplexity: " << calculate_perplexity() << endl;
            }
        }
    }
}

float LDA::calculate_perplexity() {
    float log_likelihood = 0;
    for (size_t d = 0; d < docs.size(); ++d) {
        for (size_t n = 0; n < docs[d].size(); ++n) {
            // is it important that likelihood is a float and not a double?
            auto thisIsADouble = 0.0;
            auto thisIsAFloat = 0.0f;
            auto thisIsAnInt = 0;

            float likelihood = 0;
            size_t term_id = docs[d][n];
            size_t N = docs[d].size();
            // inside every for loop there is a higher order function wanting to get out.
            // here you can use std::accumulate http://en.cppreference.com/w/cpp/algorithm/accumulate, 
            // the surrounding for loop also looks like accumulate.
            for (size_t k = 0; k < K; ++k) {
                likelihood += ((CDK[d][k] + alpha) * (CKW[k][term_id] + beta))
                    / ((K * alpha + N) * (CK[k] + V * beta));
            }
            log_likelihood += log(likelihood);
        }
    }
    // took me a while here to figure out that total_num_of_words is a member variable, use _ prefix
    return exp(-log_likelihood / total_num_of_words);
}

vector<vector<float>> LDA::getDocTopicMatrix() const {
    auto theta = vector<vector<float>>(docs.size(), vector<float>(K));
    for (auto d = 0; d < docs.size(); ++d) {
        auto N = docs[d].size();
        // there is a std::transform here, see http://en.cppreference.com/w/cpp/algorithm/transform
        for (size_t k = 0; k < K; ++k) {
            theta[d][k] = (CDK[d][k] + alpha) / (N + K * alpha);
        }
    }
    return theta;
}

// this is what you need
template<class T>
vector<vector<T>> build2dvector(int x, int y)
{
    return vector<vector<T>>(x, vector<T>(y));
}

vector<vector<float>> LDA::getTopicTermMatrix() const {
    auto phi = build2dvector<float>(K, V);
    // definitely another std::transform here, try to make two nested std::transforms, that could be fun
    for (size_t k = 0; k < K; ++k) {
        for (size_t v = 0; v < V; ++v) {
            phi[k][v] = (CKW[k][v] + beta) / (CK[k] + V * beta);
        }
    }
    return phi;
}

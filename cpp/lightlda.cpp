// Copyright 2018 Arnab Bhadury
#include "cpp/lightlda.h"

namespace fastlda {
LightLDA::LightLDA(const vector<vector<size_t>> &docs, const size_t V, const size_t K,
                   const float alpha, const float beta) : LDA(docs, V, K, alpha, beta),
                        beta_samples(vector<size_t>(K)),
                        term_samples(vector<vector<size_t>>(V, vector<size_t>(K))),
                        sample_counts(vector<size_t>(V + 1)),
                        uniform_topic(uniform_int_distribution<>(0, K - 1)),
                        u01(uniform_real_distribution<float>(0, 1)), beta_sum(0.0), term_sum(0.0) {

    build_beta_alias_table();

    for (size_t w = 0; w < V; ++w) {
        build_term_alias_table(w);
        for (size_t k = 0; k < K; ++k) {
            term_sum += CKW[k][w] / (CK[k] + V * beta);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        beta_sum += (beta / CK[k] + V * beta);
    }
}

void LightLDA::build_beta_alias_table() {
    vector<float> p(K);
    for (size_t k = 0; k < K; ++k) {
        p[k] = beta / (CK[k] + V * beta);
    }
    AliasTable beta_alias_table(p);
    for (size_t k = 0; k < K; ++k) {
        beta_samples[k] = beta_alias_table.get_alias_sample(generator);
    }
}

void LightLDA::build_term_alias_table(size_t w) {
    vector<float> p(K);
    for (size_t k = 0; k < K; ++k) {
        p[k] = CKW[k][w] / (CK[k] + V * beta);
    }
    AliasTable term_alias_table(p);
    for (size_t k = 0; k < K; ++k) {
        term_samples[w][k] = term_alias_table.get_alias_sample(generator);;
    }
}

void LightLDA::estimate(size_t num_iterations, size_t num_mh_steps, bool calc_perp) {
    for (size_t iter = 0; iter< num_iterations; ++iter) {
        cout << "Iteration " << iter << endl;
        size_t sample_count = 0;
        size_t accept_count = 0;
        for (size_t d = 0; d < docs.size(); ++d) {
            size_t N = docs[d].size();
            uniform_int_distribution<> doc_dist(0, N - 1);
            for (size_t n = 0; n < N; ++n) {
                for (size_t m = 0; m < num_mh_steps; ++m) {
                    sample_count++;

                    size_t topic_id = Z[d][n];
                    size_t term_id = docs[d][n];

                    size_t proposed_topic;
                    float accept = 0.0;

                    // ignore current count
                    CDK[d][topic_id]--;
                    CKW[topic_id][term_id]--;
                    CK[topic_id]--;

                    uniform_int_distribution<> coin(0, 1); // coin flip
                    auto proposal = coin(generator);
                    if (proposal == 0) {

                        float u = u01(generator) * (N - 1 + K * alpha);
                        if (u < N - 1) {
                            size_t index = doc_dist(generator);
                            proposed_topic = Z[d][index];
                        } else {
                            proposed_topic = uniform_topic(generator);
                        }
                        // MH Probability

                        accept = (CKW[proposed_topic][term_id] + beta)
                                 / (CKW[topic_id][term_id] + beta);
                        accept *= (CK[topic_id] + V * beta) / (CK[proposed_topic] + V * beta);
                        accept *= (CDK[d][proposed_topic] + alpha) / (CDK[d][topic_id] + alpha);
                        accept *= (CDK[d][topic_id] + alpha + 1)
                                  / (CDK[d][proposed_topic] + alpha + 1);
                    } else {
                        if (sample_counts[V] >= K) {
                            build_beta_alias_table();
                            sample_counts[V] = 0;
                        }
                        if (sample_counts[term_id] >= K) {
                            build_term_alias_table(term_id);
                            sample_counts[term_id] = 0;
                        }
                        float u = u01(generator) * (beta_sum + term_sum);
                        if (u < beta_sum) {
                            proposed_topic = beta_samples[sample_counts[V]];
                            sample_counts[V]++;
                        } else {
                            proposed_topic = term_samples[term_id][sample_counts[term_id]];
                            sample_counts[term_id]++;
                        }
                        accept = (CDK[d][proposed_topic] + alpha) / (CDK[d][topic_id] + alpha);
                        accept *= (CKW[proposed_topic][term_id] + beta)
                                  / (CKW[topic_id][term_id] + beta);
                        accept *= (CK[topic_id] + V * beta) / (CK[proposed_topic] + V * beta);
                        accept *= (CKW[topic_id][term_id] + beta + 1)
                                  / (CKW[proposed_topic][term_id] + beta + 1);
                        accept *= (CK[proposed_topic] + 1 + V * beta)
                                  / (CK[topic_id] + 1 + V * beta);
                    }
                    if (u01(generator) >= accept) {
                        // Reject proposed sample
                        proposed_topic = topic_id;
                    } else {
                        accept_count++;
                        beta_sum -= (beta / (CK[topic_id] + V * beta));
                        beta_sum += (beta / (CK[proposed_topic] + V * beta));  // fix this
                        term_sum -= (CKW[topic_id][term_id] / (CK[topic_id] + V * beta));
                        term_sum += (CKW[proposed_topic][term_id]
                                    / (CK[proposed_topic] + V * beta));
                    }
                    CDK[d][proposed_topic]++;
                    CKW[proposed_topic][term_id]++;
                    CK[proposed_topic]++;
                    Z[d][n] = proposed_topic;
                }
            }
        }
        cout << "MH Acceptance Rate: " << float(accept_count) / sample_count << endl;
        if (calc_perp) {
            if (iter && (iter % 10 == 0 || iter == num_iterations - 1)) {
                cout << "Perplexity: " << calculate_perplexity() << endl;
            }
        }

    }
}
}  // fastlda

// Copyright 2018 Arnab Bhadury
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

#include "cpp/alias_table.h"
#include "cpp/cgs_lda.h"
#include "cpp/lda_inference.h"
#include "cpp/lightlda.h"

namespace py = pybind11;

namespace fastlda {

PYBIND11_MODULE(fastlda, m) {
    py::class_<CGS_LDA>(m, "CGS_LDA")
       .def(py::init<const vector<vector<size_t>> &, const size_t &,
                     const size_t &, const float &, const float & >(),
                     R"pbdoc(
                       Latent Dirichlet Allocation with Collapsed Gibbs Sampling
                     )pbdoc")
       .def("estimate", &CGS_LDA::estimate, "Estimate LDA parameters",
            py::arg("num_iterations") = 100, py::arg("calc_perp") = false)
       .def("calculatePerplexity", &CGS_LDA::calculatePerplexity)
       .def("getTopicTermMatrix", &CGS_LDA::getTopicTermMatrix)
       .def("getDocTopicMatrix", &CGS_LDA::getDocTopicMatrix);


     py::class_<LightLDA>(m, "LightLDA")
        .def(py::init<const vector<vector<size_t>> &, const size_t &,
                      const size_t &, const float &, const float & >(),
                      R"pbdoc(
                        Latent Dirichlet Allocation with Alias Sampling
                      )pbdoc")
        .def("estimate", &LightLDA::estimate, "Estimate LDA parameters",
             py::arg("num_iterations") = 100, py::arg("num_mh_steps") = 4,
             py::arg("calc_perp") = false)
        .def("calculatePerplexity", &LightLDA::calculatePerplexity)
        .def("getTopicTermMatrix", &LightLDA::getTopicTermMatrix)
        .def("getDocTopicMatrix", &LightLDA::getDocTopicMatrix);


    py::class_<LDAInference>(m, "LDAInference")
        .def(py::init<const vector<vector<float>> &, const float & >(),
            "Inference module for Latent Dirichlet Allocation")
        .def("infer", &LDAInference::infer,
            "Infer latent topics for a document");

    py::class_<AliasTable>(m, "AliasTable")
        .def(py::init<const vector<float> &>(),
            "Alias Table module")
        .def("get_alias_sample", &AliasTable::getAliasSample);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace fastlda

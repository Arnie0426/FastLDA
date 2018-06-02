#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "lda_inference.h"
#include "lightlda.h"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(fastlda, m) {
//    py::class_<LDA>(m, "LDA")
//        .def(py::init<const vector<vector<size_t>> &, const size_t &,
//                      const size_t &, const float &, const float & >(),
//                      R"pbdoc(
//                          Latent Dirichlet Allocation
//                      )pbdoc")
//        .def("estimate", &LDA::estimate, "Estimate LDA parameters",
//             py::arg("num_iterations") = 100, py::arg("calc_perp") = false)
//        .def("calculate_perplexity", &LDA::calculate_perplexity)
//        .def("getTopicTermMatrix", &LDA::getTopicTermMatrix)
//        .def("getDocTopicMatrix", &LDA::getDocTopicMatrix);
//
//    m.doc() = R"pbdoc(
//        Latent Dirichlet Allocation training module exposed to Python
//        -----------------------
//           estimate
//           getTopicTermMatrix
//           getDocTopicMatrix
//    )pbdoc";

//     py::class_<CGS_LDA>(m, "CGS_LDA")
//        .def(py::init<const vector<vector<size_t>> &, const size_t &,
//                      const size_t &, const float &, const float & >(),
//                      R"pbdoc(
//                          Latent Dirichlet Allocation
//                      )pbdoc")
//        .def("estimate", &CGS_LDA::estimate, "Estimate LDA parameters",
//             py::arg("num_iterations") = 100, py::arg("calc_perp") = false)
//        .def("calculate_perplexity", &CGS_LDA::calculate_perplexity)
//        .def("getTopicTermMatrix", &CGS_LDA::getTopicTermMatrix)
//        .def("getDocTopicMatrix", &CGS_LDA::getDocTopicMatrix);


     py::class_<LightLDA>(m, "LightLDA")
        .def(py::init<const vector<vector<size_t>> &, const size_t &,
                      const size_t &, const float &, const float & >(),
                      R"pbdoc(
                          Latent Dirichlet Allocation with Alias Sampling
                      )pbdoc")
        .def("estimate", &LightLDA::estimate, "Estimate LDA parameters",
             py::arg("num_iterations") = 100, py::arg("num_mh_steps") = 4,
             py::arg("calc_perp") = false)
        .def("calculate_perplexity", &LightLDA::calculate_perplexity)
        .def("getTopicTermMatrix", &LightLDA::getTopicTermMatrix)
        .def("getDocTopicMatrix", &LightLDA::getDocTopicMatrix);

    m.doc() = R"pbdoc(
        CGS Latent Dirichlet Allocation training module exposed to Python
        -----------------------
           estimate
           getTopicTermMatrix
           getDocTopicMatrix
    )pbdoc";

    py::class_<LDA_Inference>(m, "LDA_Inference")
        .def(py::init<const vector<vector<float>> &, const float & >(),
            "Inference module for Latent Dirichlet Allocation")
        .def("infer", &LDA_Inference::infer,
            "Infer latent topics for a document");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

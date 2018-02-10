#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "lda.h"
namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(fastlda, m) {
    py::class_<LDA>(m, "LDA")
        .def(py::init<const vector<vector<size_t>> &, const size_t &,
                      const size_t &, const float &, const float & >(),
                      R"pbdoc(
                          Latent Dirichlet Allocation
                      )pbdoc")
        .def("estimate", &LDA::estimate)
        .def("calculate_perplexity", &LDA::calculate_perplexity)
        .def("getTopicTermMatrix", &LDA::getTopicTermMatrix)
        .def("getDocTopicMatrix", &LDA::getDocTopicMatrix);

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    //
    // m.def("add", &add, R"pbdoc(
    //     Add two numbers
    //
    //     Some other explanation about the add function.
    // )pbdoc");
    //
    // m.def("multiply", &mul, R"pbdoc(Multiplies two numbers)");
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

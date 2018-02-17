# FastLDA -- A minimal and fast implementation of Latent Dirichlet Allocation

## WIP.

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1


## Installation

1. Fork/clone this repo.

2. In fastlda/ 
```bash
2. pip install --upgrade .
```
should install all the dependencies needed.

3. Run
```bash
python python/lda_example.py
```
for an end-to-end LDA run (from training to inference using NIPS dataset from https://archive.ics.uci.edu/ml/datasets/bag+of+words).


## License

MIT

This code is written in C++ with python modules exposed using pybind11.

# FastLDA -- A minimal and fast implementation of Latent Dirichlet Allocation

## Still a work in progress, but currently fully usable!

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1


## Installation

### Building from source

1. Fork/clone this repo.

2. In fastlda/ please run
```bash
git submodule init
git submodule update
pip install --upgrade .
```
This should install all the dependencies needed.

3. Run
```bash
python python/lda_example.py
```
for an end-to-end LDA run (from training to inference using NIPS dataset from https://archive.ics.uci.edu/ml/datasets/bag+of+words).

### Alternatives

- [ ] Pypi version soon to come!

## Usage

A Python class has been exposed with all the relevant training/inference functions.

For an example, see: [this simple example](https://github.com/Arnie0426/FastLDA/blob/master/python/lda_example.py#L26-L51)

## License

MIT

This code is written in C++ with python modules exposed using pybind11.

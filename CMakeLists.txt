cmake_minimum_required(VERSION 3.17)
project(bayesTextClassifier)

set(CMAKE_CXX_STANDARD 11)

add_executable(bayesTextClassifier main.cpp BayesClassifier.h)

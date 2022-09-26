# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "loss.cpp":
    pass

cdef extern from "loss.h":
    cdef cppclass Util:
        double calculateSELoss(double* X, double* y, double split, int nSamples);
        double calculateWELoss(double* X, double* y, double split, int nSamples, int nClasses);
        

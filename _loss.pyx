cimport _loss
cimport numpy as cnp

cdef class PyLoss:
    cdef Util* c_util

    def __cinit__(self):
        self.c_util = new Util()

    def __dealloc__(self):
        del self.c_util

    def calc_we(self, double[:] X, double[:,:] y, double split, int nSamples, int nClasses):
        return self.c_util.calculateWELoss(&X[0], &y[0,0], split, nSamples, nClasses)

    def calc_se(self, double[:] X, double[:] y, double split, int nSamples):
        return self.c_util.calculateSELoss(&X[0], &y[0], split, nSamples)
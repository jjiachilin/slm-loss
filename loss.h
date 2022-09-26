#ifndef UTIL_H
#define UTIL_H

#include <vector>
using namespace std;

class Util 
{
    public:

        double calculateSELoss(double* X, double* y, double split, int nSamples);
        double calculateWELoss(double* X, double* y, double split, int nSamples, int nClasses);
};

#endif
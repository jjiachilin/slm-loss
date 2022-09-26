#include <vector> 
#include <cmath>
#include "loss.h"

using namespace std;

// returns squared error loss
double Util::calculateSELoss(double* X, double* y, double split, int nSamples)
{
    int leftySize = 0;
    int rightySize = 0;
    double leftySum = 0;
    double rightySum = 0;

    for (int j = 0; j < nSamples; ++j)
    {
        if (X[j] <= split)
        {
            ++leftySize;
            leftySum += y[j];
        }
        else
        {
            ++rightySize;
            rightySum += y[j];
        }
    }

    double leftyMean = leftySum / leftySize;
    double rightyMean = rightySum / rightySize;
    double leftyH = 0;
    double rightyH = 0;

    for (int j = 0; j < nSamples; ++j)
    {
        if (X[j] <= split)
        {
            leftyH += (y[j] - leftyMean) * (y[j] - leftyMean);
        }
        else
        {
            rightyH += (y[j] - rightyMean) * (y[j] - rightyMean);
        }
    }

    return (leftySize / (double)nSamples) * leftyH + (rightySize / (double)nSamples) * rightyH;
}

// returns weighted entropy loss
// y is one hot encoded ie. shape is (nSamples, nClasses)
double Util::calculateWELoss(double* X, double* y, double split, int nSamples, int nClasses)
{
    double leftyProbs[100];
    double rightyProbs[100];

    int leftySize = 0;
    int rightySize = 0;

    for (int i = 0; i < nClasses; ++i)
    {
        leftyProbs[i] = 0;
        rightyProbs[i] = 0;
    }

    for (int j = 0; j < nSamples; ++j)
    {
        int maxIdx = 0;
        if (X[j] <= split)
        {
            ++leftySize;
            for (int k = 1; k < nClasses; ++k)
            {
                if (y[j*nClasses+k] > y[j*nClasses+maxIdx])
                {
                    maxIdx = k;
                }
            }
            ++leftyProbs[maxIdx];
        }
        else
        {
            ++rightySize;
            for (int k = 1; k < nClasses; ++k)
            {
                if (y[j*nClasses+k] > y[j*nClasses+maxIdx])
                {
                    maxIdx = k;
                }
            }
            ++rightyProbs[maxIdx];
        }
    }

    for (int j = 0; j < nClasses; ++j)
    {
        leftyProbs[j] /= (double)leftySize;
        rightyProbs[j] /= (double)rightySize;
    }

    double rightyE = 0;
    double leftyE = 0;

    for (int j = 0; j < nClasses; ++j)
    {
        if (rightyProbs[j] > 0)
            rightyE -= rightyProbs[j] * log(rightyProbs[j]);
        if (leftyProbs[j] > 0)
            leftyE -= leftyProbs[j] * log(leftyProbs[j]);
    }

    rightyE /= log(nClasses);
    leftyE /= log(nClasses);

    return (leftySize / (double)nSamples) * leftyE + (rightySize / (double)nSamples) * rightyE;
}

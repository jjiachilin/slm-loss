# slm-loss

## Notes
X and y should have data type double\
X and y should be one dimensional for squared error loss (ie. X.shape = y.shape = (nSamples,))\
X should be one dimensional and y should be two dimensional and one hot encoded for weighted entropy loss (ie. y.shape = (nSamples, nClasses))

## Building
Make sure cython is installed in your environment then
```
python setup.py build_ext --inplace
```

## Usage
```
from _loss import PyLoss

loss = PyLoss()
X = ...
y = ...
split = ...
nSamples = X.shape[0]
loss.calc_se(X, y, split, nSamples)
```
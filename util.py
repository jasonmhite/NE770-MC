import numpy as np

__all__ = ['Bounds']

class Bounds(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.upper))
        tmin = bool(np.all(x >= self.lower))

        return(tmax and tmin)

    def check(self, x):
        tmax = bool(np.all(x <= self.upper))
        tmin = bool(np.all(x >= self.lower))

        return(tmax and tmin)

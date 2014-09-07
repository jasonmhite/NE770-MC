from __future__ import division
import numpy as np
from .sampling import Sampler

__all__ = ['RejectionSampler']

class RejectionSampler(Sampler):

    """Rejection sampler. Will adaptively select the batch size."""

    def __init__(self, pdf, tallyfxn, lower, upper, hmax):
        Sampler.__init__(self, tallyfxn, pdf)
        self.hmax = hmax
        self.lower = lower
        self.upper = upper

    def _draw_samples(self, N):
        Nh = N
        prop = 0
        S = np.array([])
        eff = 0.9

        while len(S) < N:
            xi1 = np.random.rand(np.ceil(N / eff))
            xi2 = np.random.rand(np.ceil(N / eff))

            X = self.lower + xi1 * (self.upper - self.lower)
            htilde = self.hmax * xi2

            Z = htilde <= X

            x_accept = X[Z]

            if len(x_accept) > Nh:
                x_accept = x_accept[:Nh]
                prop += np.argwhere(Z).flatten()[Nh]
            else:
                prop += len(X)

            S = np.hstack((S, x_accept))
            Nh -= len(x_accept)

            eff = max(0.05, (N - Nh) / prop)

        return(S, prop)

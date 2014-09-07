import numpy as np
from .sampling import Sampler

__all__ = ['RejectionSampler']

class RejectionSampler(Sampler):

    """Rejection sampler"""

    def __init__(self, pdf, tallyfxn, lower, upper, hmax):
        Sampler.__init__(self, tallyfxn, pdf)
        self.hmax = hmax
        self.lower = lower
        self.upper = upper

    def _draw_samples(self, N):
        Nh = N
        prop = 0
        S = np.array([])
        while len(S) < N:
            # TODO: More efficient to use larger batch size
            eps1 = np.random.rand(Nh)
            eps2 = np.random.rand(Nh)

            htilde = self.hmax * eps2
            xi = self.lower + eps1 * (self.upper - self.lower)

            x_accept = xi[htilde <= xi]

            S = np.hstack((S, x_accept))
            prop += Nh
            Nh -= len(x_accept)

        return(S, prop)

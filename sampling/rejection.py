from __future__ import division
import numpy as np
import scipy.optimize as sop
import scipy.integrate as sin
from .sampling import Sampler

__all__ = ['RejectionSampler', 'MultiRegionRejectionSampler']


class RejectionSampler(Sampler):

    """Rejection sampler. Will adaptively select the batch size."""

    def __init__(self, pdf, lower, upper, hmax=None):
        Sampler.__init__(self)
        self.pdf = pdf
        if hmax is None:
            bounds = np.array([[lower, upper]])
            x0 = np.atleast_1d((upper - lower) / 2.)
            r = sop.minimize(
                lambda x: np.atleast_1d(-1. * pdf(x)),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 50},
            )
            self.hmax = -1. * r.fun[0]
        else:
            self.hmax = hmax
        self.lower = lower
        self.upper = upper
        self.prop = 0
        self.accept = 0

    def _draw_samples(self, N):
        Nh = N
        S = np.array([])
        eff = 0.9

        prop = 0

        while len(S) < N:
            xi1 = np.random.rand(np.ceil(N / eff))
            xi2 = np.random.rand(np.ceil(N / eff))

            X = self.lower + xi1 * (self.upper - self.lower)
            P_X = self.pdf(X)
            htilde = self.hmax * xi2

            Z = htilde <= P_X

            x_accept = X[Z]

            if len(x_accept) > Nh:
                x_accept = x_accept[:Nh]
                prop += np.argwhere(Z).flatten()[Nh] + 1
            else:
                prop += len(X)

            self.accept += len(x_accept)

            S = np.hstack((S, x_accept))
            Nh -= len(x_accept)

            eff = max(0.05, (N - Nh) / prop)

        self.prop += prop

        return(S)

    @property
    def e(self):
        return(self.accept / self.prop)


class MultiRegionRejectionSampler(Sampler):

    """See class name..."""

    def __init__(self, pdf, regions):
        Sampler.__init__(self)

        self.samplers = []
        self.weights = []
        self.n_regions = len(regions)

        for lower, upper in regions:
            S = RejectionSampler(pdf, lower, upper)
            S_H, err = sin.quad(pdf, lower, upper)

            self.samplers.append(S)
            self.weights.append(S_H)

        self.weights = np.array(self.weights)
        # Adjust weights to sum to 1 for numeric noise
        self.weights /= self.weights.sum()

    def _draw_samples(self, N):
        ns = np.random.choice(
            np.arange(self.n_regions),
            N,
            p=self.weights,
        )
        ns = [len(np.extract(ns == i, ns)) for i in xrange(self.n_regions)]

        S = np.array([])

        for (Si, n) in zip(self.samplers, ns):
            S = np.hstack((S, Si._draw_samples(n)))

        # Not strictly necessary, but shuffle things up
        np.random.shuffle(S)

        return(S)

    @property
    def prop(self):
        p = 0
        for Si in self.samplers:
            p += Si.prop

        return(p)

    @property
    def ei(self):
        return(np.array([Si.e for Si in self.samplers]))

    @property
    def e(self):
        return(1. / ((self.weights / self.ei).sum()))

    @property
    def h(self):
        return([a.hmax for a in self.samplers])

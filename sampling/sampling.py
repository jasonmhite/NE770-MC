import numpy as np
import abc

__all__ = ['Sampler']


class Sampler(object):

    """Monte Carlo sampler base class"""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.N = 0  # Total samples drawn
        self.X = np.array([])

    @abc.abstractmethod
    def _draw_samples(self, N):
        return NotImplemented

    def run(self, N, reset=False):
        if reset:
            self.N = 0
            self.X = np.array([])
        X_new = self._draw_samples(N)
        self.X = np.hstack((self.X, X_new))
        self.N += len(X_new)

        mu = X_new.mean()
        var = X_new.var()
        err = var / mu

        return(mu, var, err)

    # These will need to be made into @abstractproperties
    # for tally-weighted values.

    @property
    def mu(self):
        return(self.X.mean())

    @property
    def var(self):
        return(self.X.var())

    @property
    def err(self):
        return(self.var / self.N)

import numpy as np
import abc

__all__ = ['Sampler']

class Sampler(object):

    """Monte Carlo sampler base class"""

    __metaclass__ = abc.ABCMeta

    def __init__(self, tallyfxn, pdf):
        self.N = 0 # Total samples drawn
        self.S = 0 # Successes
        self.tally = tallyfxn
        self.pdf = pdf

    @abc.abstractmethod
    def _draw_samples(self, N):
        return NotImplemented

    def run(self, N, reset=False):
        if reset:
            self.N = 0
            self.S = 0
        X = self._draw_samples(N)
        self.N += len(X)
        self.S += self.tally(X)

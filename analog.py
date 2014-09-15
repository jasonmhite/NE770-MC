from .sampling import Sampler
import numpy as np

__all__ = ['AnalogIntegrator']

class AnalogIntegrator(Sampler):

    """Analog Monte Carlo sampler"""

    def __init__(self, fx, lower, upper):
        Sampler.__init__(self)
        self.fx = fx
        self.lower = lower
        self.upper = upper
        self.scale = upper - lower

    def _draw_samples(self, N):
        xi = self.scale * np.random.rand(N) + self.lower
        S = self.scale * self.fx(xi)

        return(S)

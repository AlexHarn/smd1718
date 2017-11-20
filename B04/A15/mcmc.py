import numpy as np
import scipy.stats as scs


class MCMC(object):
    def __init__(self, step_size=1., loc=0., scale=1.):
        """
        Sample from a 1D gaussian PDF with uniform step proposal.

        Parameters
        ----------
        loc, scale : float
            Mean and standard deviation for the gaus to sample from.
        step_size = float
            Step sized used symmetrically around the current step to propose
            the next one from a uniform PDF in ``[-step_size, step_size]``.
        """
        self.step_size = step_size
        self.pdf = scs.norm(loc, scale).pdf

    def _propose_step(self, xi):
        """
        Calculate the next proposed step from the current one from the
        step proposal PDF (here: uniform).

        Parameters
        ----------
        xi : float
            Current step from which the next positions is calculated.

        Returns
        -------
        xj : float
            Next proposed step.
        """
        return xi + np.random.uniform(-1, 1)*self.step_size

    def _accept_step(self, xi, xj):
        """
        Decide wether to accept the next step or not using the
        Metropolis-Hastings detailed balance condition.

        Parameters
        ----------
        xi : float
            Current step from which the next positions is calculated.
        xj : float
            Next proposed step.

        Returns
        -------
        acc : bool
            ``True``if the next step is accepted, ``False`` if not.
        """
        return np.random.uniform() <= self.pdf(xj)/self.pdf(xi)

    def sample(self, x0, n=1):
        """
        Sample ``n`` points from the gaussian PDF using the MCMC algorithm.

        Parameters
        ----------
        x0 : float
            Start value where the Markov chain is started.
        n : int
            How many samples to create.

        Returns
        -------
        x : array-like
            Created sample points. Has length ``n``.
        """
        x = np.empty(n, dtype=float)
        x[0] = x0
        for i in range(1, n):
            proposed = self._propose_step(x[i-1])
            if self._accept_step(x[i-1], proposed):
                x[i] = proposed
            else:
                x[i] = x[i-1]

        return x

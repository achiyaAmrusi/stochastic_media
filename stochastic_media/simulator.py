import numpy as np
import xarray as xr
from stochastic_media.sample import LayeredSample

class SLMSimulator:
    """
    Simulator for stochastic layered media (SLM) attenuation.

    This class models radiation transport (or wave attenuation) through
    a one-dimensional, statistically mixed medium consisting of two materials
    arranged in microscopic layers.

    The stochasticity is introduced by randomly generating the number of
    sub-layers of each material in a given realization based on their
    volume fractions. Each realization represents one possible microscopic
    configuration of the medium.

    Attributes
    ----------
    sample : LayeredSample
        The layered sample object containing geometry and material data.

    Methods
    -------
    simulate_attenuation(size)
        Generate multiple stochastic realizations and compute their attenuation.
    get_statistics(attenuation)
        Compute mean and standard deviation of the simulated attenuation.
    """

    def __init__(self, sample:LayeredSample):
        self.sample = sample

    def simulate_attenuation(self, size):
        """
        Simulate the attenuation through a stochastic layered medium.

        Parameters
        ----------
        size : int
            Number of random realizations (samples) to generate.

        Returns
        -------
        attenuation : xr.DataArray
            Array of attenuation values for each realization.
        """
        n = np.ceil(self.sample.width/self.sample.dx)
        k_samples = xr.DataArray(np.random.binomial(n, self.sample.volume_fraction_1, size=size), coords={'samples':range(size)})
        sigma_samples = (k_samples * self.sample.material_1.xs +(n-k_samples) * self.sample.material_2.xs) / n
        attenuation = np.exp(-sigma_samples * self.sample.width)
        return attenuation

    def get_statistics(self, attenuation):
        """
        Compute simple statistics (mean and standard deviation) of the attenuation.

        Parameters
        ----------
        attenuation : xr.DataArray
            Array of attenuation values (output of simulate_attenuation).

        Returns
        -------
        dict
            Dictionary with 'mean' and 'std' of attenuation.
        """
        return {
            "mean": attenuation.mean('samples'),
            "std": attenuation.std('samples')
        }

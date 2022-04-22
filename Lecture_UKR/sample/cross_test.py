import unittest
from ukr_jax import UKR as ukr_jax
from ukr_torch import UKR as ukr_torch
from Lecture_UKR.data import create_kura

import numpy as np


class TestUKR(unittest.TestCase):
    def test_UKR(self):
        epoch = 1000
        sigma = 0.2
        eta = 100
        latent_dim = 2
        nb_samples = 100
        seed = 4
        np.random.seed(seed)

        X = create_kura(nb_samples)
        Zinit = 0.2 * sigma * np.random.rand(nb_samples, latent_dim) - 0.1 * sigma

        ukr_j = ukr_jax(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_j.fit(epoch, eta)

        ukr_t = ukr_torch(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_t.fit(epoch, eta)

        print("You are executing ", __file__)

        np.testing.assert_allclose(
            ukr_j.history['z'],
            ukr_t.history['z'],
            rtol=1e-05,
            atol=1e-08)
        np.testing.assert_allclose(
            ukr_j.history['f'],
            ukr_t.history['f'],
            rtol=1e-05,
            atol=1e-08)


if __name__ == "__main__":
    unittest.main()

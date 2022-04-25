import unittest
from ukr_jax import UKR as ukr_jax
from ukr_torch import UKR as ukr_torch
from ukr_tf import UKR as ukr_tf
from ukr_numpy import UKR as ukr_numpy
from Lecture_UKR.data import create_kura

import numpy as np


class TestUKR(unittest.TestCase):
    def test_UKR(self):
        epoch = 100
        sigma = 0.2
        eta = 100
        latent_dim = 2
        nb_samples = 100
        seed = 4
        np.random.seed(seed)

        X = create_kura(nb_samples)
        Zinit = 0.2 * sigma * np.random.rand(nb_samples, latent_dim) - 0.1 * sigma

        print('ukr by torch')
        ukr_t = ukr_torch(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_t.fit(epoch, eta)

        print('ukr by numpy')
        ukr_num = ukr_numpy(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_num.fit(epoch, eta)

        print('ukr by tensorflow')
        ukr_tf_ = ukr_tf(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_tf_.fit(epoch, eta)

        print('ukr by jax')
        ukr_j = ukr_jax(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_j.fit(epoch, eta)



        print("You are executing ", __file__)

        np.testing.assert_allclose(
            ukr_num.history['z'],
            ukr_tf_.history['z'],
            rtol=1e-05,
            atol=1e-08)
        np.testing.assert_allclose(
            ukr_num.history['f'],
            ukr_tf_.history['f'],
            rtol=1e-05,
            atol=1e-08)


if __name__ == "__main__":
    unittest.main()

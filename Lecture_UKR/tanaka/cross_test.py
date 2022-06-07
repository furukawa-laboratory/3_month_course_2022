#sampleフォルダのukr_torchとクロステストするプログラム

import unittest
from Lecture_UKR.data import create_kura
from sample.ukr_torch import UKR as ukr_sample

#作ったプログラムを継承させてね(以下例)
from ukr import UKR as ukr

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

        #サンプルプログラム
        ukr_sam = ukr_sample(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_sam.fit(epoch, eta)

        #作ったプログラム
        ukr_mine = ukr(X, latent_dim, sigma, prior='random', Zinit=Zinit)
        ukr_mine.fit(epoch, eta)

        print("You are executing ", __file__)

        np.testing.assert_allclose(
            ukr_sam.history['z'],
            ukr_mine.history['z'],
            rtol=1e-05,
            atol=1e-08)
        np.testing.assert_allclose(
            ukr_sam.history['f'],
            ukr_mine.history['f'],
            rtol=1e-05,
            atol=1e-08)


if __name__ == "__main__":
    unittest.main()

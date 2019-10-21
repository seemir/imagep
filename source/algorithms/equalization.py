# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from .image_process import ImageProcess
import numpy as np


class Equalization(ImageProcess):

    def __init__(self, image: np.ndarray = None, bins=256):
        """
        Contrast adjustment using the image's histogram. The returned image will be in the
        form of an 8-bit (2^3) integer array with 255 as white and 0 as black.

        Parameters
        ----------
        image       : np.ndarray
                      Image to apply contrast adjustment on. If color image the last dimension
                      is considered color value, i.e. as RGB values.
        bins        : int
                      number of bins for histogram

        """
        super().__init__(image=image, bins=bins)

    def equalization(self):
        """
        Implementation of the histogram equalization algorithm

        Returns
        -------
        Out         : np.ndarray()
                      Image cumulative histogram with 256 bins

        """
        image = self.image.astype(int)
        m, n = image.shape
        k = self.bins
        cum_hist = self.cum_hist(image, list(range(k + 1)))
        mm = m * n
        for i in range(m):
            for j in range(n):
                a = image[i, j]
                b = cum_hist[a] * (k - 1) / mm
                image[i, j] = b
        return image

    def compare_images(self):
        """
        Compares side-by-side the original and equalized image

        """
        self._compare_images(self.image.ravel(), self.image, self.equalization().ravel(),
                             self.equalization(), title="Histogram Equalization", bins=self.bins)

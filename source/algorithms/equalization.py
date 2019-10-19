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

    def histogram(self):
        """
        Method for producing Image histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image histogram with 256 bins

        """
        # Scratch implementation
        # -------------------------------------------------------
        # m, n, *_ = self.image.shape
        # hist = np.zeros(self.bins)
        # for i in range(m):
        #     for j in range(n):
        #         hist[int(self.image[i, j])] += 1
        # return hist
        #
        # Numpy-Implementation
        # -------------------------------------------------------
        return np.histogram(self.image.ravel(), self.bins)

    def cum_hist(self):
        """
        Method for producing Image cumulative histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image cumulative histogram with 256 bins

        """
        # Scratch implementation
        # # -------------------------------------------------------
        # cum_hist = np.zeros(self.bins)
        # hist = self.histogram()
        # cum_hist[0] = hist[0]
        # for i in range(self.bins - 1):
        #     cum_hist[i + 1] = cum_hist[i] + hist[i + 1]
        # return cum_hist
        #
        # Numpy-Implementation
        # -------------------------------------------------------
        return np.cumsum(self.histogram()[0])

    def equalization(self):
        """
        Implementation of the histogram equalization algorithm

        Returns
        -------
        Out         : np.ndarray()
                      Image cumulative histogram with 256 bins

        """
        image = self.image.copy()
        cum_hist = self.cum_hist()
        m, n, *_ = image.shape
        k = self.bins - 1
        mm = m * n
        for i in range(m):
            for j in range(n):
                a = int(image[i, j])
                b = cum_hist[a] * (k - 1) / mm
                image[i, j] = b
        return image

    def compare_images(self):
        """
        Compares side-by-side the original and equalized image

        """
        self._compare_images(self.image.ravel(), self.image, self.equalization().ravel(),
                             self.equalization(), title="Histogram Equalization", bins=self.bins)

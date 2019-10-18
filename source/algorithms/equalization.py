# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

import matplotlib.pyplot as plt
from warnings import warn
from skimage import color
import numpy as np


class Equalization:

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
        try:
            condition = len(image[0, 0]) != 3
        except TypeError:
            condition = True

        if len(image.shape) > 3 and image.shape[-1] in (3, 4):
            warn("otsu expects grayscale images; image shape '{}' looks like an RGB image."
                 " Continuing with gray-scaled version of input image".format(image.shape))
            self.image = color.rgb2gray(image) if condition else color.rgb2gray(image.mean(axis=2))
        else:
            self.image = image if condition else image.mean(axis=2)
        self.bins = bins

    def histogram(self):
        """
        Method for producing Image histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image histogram with 256 bins

        """
        m, n, *_ = self.image.shape
        hist = np.zeros(self.bins)
        for i in range(m):
            for j in range(n):
                hist[int(self.image[i, j])] += 1
        return hist

    def cum_hist(self):
        """
        Method for producing Image cumulative histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image cumulative histogram with 256 bins

        """
        cum_hist = np.zeros(self.bins)
        hist = self.histogram()
        cum_hist[0] = hist[0]
        for i in range(self.bins - 1):
            cum_hist[i + 1] = cum_hist[i] + hist[i + 1]
        return cum_hist

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
        k = self.bins
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
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.hist(self.image.ravel(), bins=self.bins, color='b')
        plt.title("Histogram Equalization")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.imshow(self.image, cmap="gray", vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 2, 3)
        plt.hist(self.equalization().ravel(), bins=self.bins, color='b')
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.imshow(self.equalization(), cmap="gray", vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

        plt.show()

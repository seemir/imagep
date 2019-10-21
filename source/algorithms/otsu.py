# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from .image_process import ImageProcess
import numpy as np


class Otsu(ImageProcess):

    def __init__(self, image: np.ndarray, bins=256):
        """
        Binarize an image using Otsu's method[1] to find optimal threshold value. The returned image
        will be in the form of an 8-bit (2^3) integer array with 255 as white and 0 as black.

        Parameters
        ----------
        image       : np.ndarray
                      Image to apply binarization on. If color image the last dimension
                      is considered color value, i.e. as RGB values.
        bins        : int
                      number of bins for histogram

        Notes
        -------
        ..[1] Nobuyuki Otsu (1979). "A threshold selection method from gray-level histograms".
              IEEE Trans. Sys. Man. Cyber. 9 (1): 62–66. doi:10.1109/TSMC.1979.4310076.

        """
        super().__init__(image=image, bins=bins)

    def otsu_threshold(self):
        """
        Finds the optimal threshold value of given image using Otsu's method

        Returns
        -------
        Out         : numeric
                      optimal threshold based on Otsu's method

        """
        hist, center = self.histogram(self.image, self.bins)
        hist = hist.astype(float)
        w_b = np.cumsum(hist)
        w_f = np.cumsum(hist[::-1])[::-1]
        sum_b = np.cumsum(hist * center[1:]) / w_b
        sum_f = (np.cumsum((hist * center[1:])[::-1]) / w_f[::-1])[::-1]
        var = w_b[:-1] * w_f[1:] * (sum_b[:-1] - sum_f[1:]) ** 2
        i = np.argmax(var)
        th = center[:-1][i]
        return th

    def binarization(self):
        """
        Binarization function

        Returns
        -------
        Out         : np.ndarray()
                      Binarized image, i.e. all pixel values are either 0 or 255

        """
        binary = self.image <= self.otsu_threshold()
        return binary.__invert__() * (self.bins - 1)

    def compare_images(self):
        """
        Compares side-by-side the original and binarized images

        """
        self._compare_images(self.image.ravel(), self.image, self.image.ravel(),
                             self.binarization(), "Otsu Threshold: " + str(self.otsu_threshold()),
                             log=True, bins=self.bins, x_line=self.otsu_threshold())

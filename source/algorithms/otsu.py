# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

import matplotlib.pyplot as plt
from warnings import warn
import numpy as np


class Otsu:

    def __init__(self, image: np.ndarray):
        """
        Binarize an image using Otsu's method[1] to find optimal threshold value. The returned image
        will be in the form of an 8-bit (2^3) integer array with 255 as white and 0 as black.

        Parameters
        ----------
        image       : np.ndarray
                      Image to apply binarization on. If color image the last dimension
                      is considered color value, i.e. as RGB values.

        Notes
        -------
        ..[1] Nobuyuki Otsu (1979). "A threshold selection method from gray-level histograms".
              IEEE Trans. Sys. Man. Cyber. 9 (1): 62–66. doi:10.1109/TSMC.1979.4310076.

        """
        if len(image.shape) > 3 and image.shape[-1] in (3, 4):
            warn("otsu expects grayscale images; image shape '{}' looks like an RGB image."
                 " Continuing with gray-scaled version of input image".format(image.shape))
            self.image = np.dot(image, [0.2989, 0.5870, 0.1140])
        else:
            self.image = image

    def histogram(self):
        """
        Method for producing Image histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image histogram with 256 bins

        """
        # Scratch-Implementation
        # -------------------------------------------------------
        # m, n, *_ = self.image.shape
        # hist = np.zeros(256)
        # image = self.image.mean(axis=2)
        # for i in range(0, m):
        #     for j in range(0, n):
        #         pixel_value = int(image[i, j])
        #         hist[pixel_value] += 1
        # return hist

        # Numpy-Implementation
        # -------------------------------------------------------
        return np.histogram(self.image.ravel(), bins=256)

    def otsu_threshold(self):
        """
        Finds the optimal threshold value of given image using Otsu's method

        Returns
        -------
        Out         : numeric
                      optimal threshold based on Otsu's method

        """
        # Scratch-Implementation
        # -------------------------------------------------------
        # hist = self.histogram()
        # total = np.sum(hist)
        # var, maximum, th = 0, 0, 0
        # sum_b, w_b = 0, 0
        # sum_t = np.dot(np.array(range(0, 256)), hist)
        #
        # for i in range(0, 256):
        #     w_b += hist[i]
        #     w_f = total - w_b
        #     if w_b == 0 or w_f == 0:
        #         continue
        #     sum_b += i * hist[i]
        #     sum_f = sum_t - sum_b
        #     var = w_b * w_f * (sum_b / w_b - sum_f / w_f) ** 2
        #     if var > maximum:
        #         maximum = var
        #         th = i
        # return th

        # Numpy-Implementation
        # -------------------------------------------------------
        hist, center = self.histogram()
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
        Threshold function

        Returns
        -------
        Out         : np.ndarray()
                      Binarized image, i.e. all pixel values are either 0 or 255

        """
        binary = self.image <= self.otsu_threshold()
        return binary.__invert__() * 255

    def compare_images(self):
        """
        Compares side-by-side the original and binarized images

        """

        plt.figure()
        for i, method in enumerate([self.image, self.binarization()]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(method)
            plt.yticks([]), plt.xticks([])
        plt.show()

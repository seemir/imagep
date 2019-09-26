# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

import matplotlib.pyplot as plt
from skimage import io
import numpy as np


class Otsu:

    @staticmethod
    def mean_image(image, axis=2):
        """
        Returns the average of the array elements along given axis.

        Parameters
        ----------
        image       : np.ndarray
                      image to apply mean function
        axis         : numeric
                      Axis or axes along which the means are computed.

        Returns
        -------
        Out         : np.ndarray()
                      image with average elements. ValueError if image is not 2D

        Notes
        -------
        ..[1] M. Sezgin & B. Sankur (2004). "Survey over image thresholding techniques and
          quantitative performance evaluation". Journal of Electronic Imaging. 13 (1): 146–165.
          doi:10.1117/1.1631315

        """
        dim = len(image.shape)

        if dim == 3:
            image = image.mean(axis)
        else:
            raise ValueError('image must be a 2D image, got {} dimensions'.format(dim))
        return image

    def __init__(self, image: np.ndarray):
        """
        Binarize an image using Otsu's method to find optimal threshold value. The returned image
        will be in the form of an 8-bit (2^3) integer array with 255 as white and 0 as black.

        Parameters
        ----------
        image       : np.ndarray
                      Image to apply binarization on. If color image the last dimension
                      is considered color value, i.e. as RGB values.

        """
        self.image = image
        self.shape = np.shape(self.image)
        self.thresh = self.otsu_threshold(self.image)

    def binarization(self):
        """
        Threshold function

        Returns
        -------
        Out         : np.ndarray()
                      Binarized image, i.e. all pixel values are either 0 or 255

        """
        m, n, *_ = self.shape
        binarised = np.zeros([m, n], dtype=np.uint8)
        image = self.mean_image(self.image)

        binarised[image < self.thresh] = 0
        binarised[image >= self.thresh] = 255
        return binarised

    def histogram(self, image):
        """
        Method for producing Image histogram with 256 bins

        Returns
        -------
        Out         : np.ndarray()
                      Image histogram with 256 bins

        """
        m, n, *_ = self.shape
        histogram = np.zeros(256)
        image = self.mean_image(image)
        for i in range(0, m):
            for j in range(0, n):
                pixel_value = int(image[i, j])
                histogram[pixel_value] += 1
        return histogram

    def otsu_threshold(self, image):
        """
        Finds the optimal threshold value of given image using Otsu's method

        Parameters
        ----------
        image       : np.ndarray
                      Image to apply Otso's method

        Returns
        -------
        Out         : numeric
                      optimal threshold based on Otsu's method

        """
        hist = self.histogram(image)
        total = np.sum(hist)
        var, maximum, th = 0, 0, 0
        sum_b, w_b = 0, 0
        sum_t = np.dot(np.array(range(0, 256)), hist)

        for i in range(0, 256):
            w_b += hist[i]
            w_f = total - w_b
            if w_b == 0 or w_f == 0:
                continue
            sum_b += i * hist[i]
            sum_f = sum_t - sum_b
            var = w_b * w_f * (sum_b / w_b - sum_f / w_f) ** 2
            if var > maximum:
                maximum = var
                th = i
        return th

    def compare_images(self):
        """
        Compares side-by-side the original and binarized images

        Returns
        -------

        """
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        io.imshow(self.image)
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2)
        io.imshow(self.binarization())
        plt.xticks([]), plt.yticks([])

        io.show()

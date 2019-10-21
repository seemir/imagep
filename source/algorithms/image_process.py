# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from warnings import warn
from skimage import color
import numpy as np


class ImageProcess(ABC):

    @abstractmethod
    def __init__(self, image: np.ndarray = None, bins: int = None):
        """
        image processing base class

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

    @staticmethod
    def histogram(image: np.ndarray = None, bins: int = None):
        """
        Method for producing Image histogram with 256 bins

        Parameters
        ----------
        see constructor

        Returns
        -------
        Out         : np.ndarray()
                      Image histogram with 256 bins

        """
        return np.histogram(image, bins)

    @staticmethod
    def cum_hist(image: np.ndarray = None, bins: (int, list) = None):
        """
        Method for producing Image cumulative histogram with 256 bins

        Parameters
        ----------
        see constructor

        Returns
        -------
        Out         : np.ndarray()
                      Image cumulative histogram with 256 bins

        """
        return np.array(np.cumsum(ImageProcess.histogram(image, bins)[0]))

    @staticmethod
    def _compare_images(top_hist, top_pic, bottom_hist, bottom_pic, title, log=False, bins=256,
                        x_line=None):
        """
        Compares side-by-side the original and equalized image

        Parameters
        ----------
        top_hist        : np.ndarray
                          first (top left) histogram to plot
        top_pic         : np.ndarray
                          top right image to display
        bottom_hist     : np.ndarray
                          second (bottom left) histogram to add to plot
        log             : bool
                          logarithmic scale on bottom left histogram
        bottom_pic      : np.ndarray
                          bottom right processed image
        title           : str
                          title to be displayed
        bins            : int
                          number of bins
        x_line            : int
                          vertical line through plot at x position

        """
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.hist(top_hist, list(range(bins)), color='b')
        plt.axvline(x_line, color='r') if x_line else None
        plt.title(title)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.imshow(top_pic, cmap="gray", vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 2, 3)
        plt.hist(bottom_hist, list(range(bins)), color='b', log=log)
        plt.axvline(x_line, color='r') if x_line else None
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.imshow(bottom_pic, cmap="gray", vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

        plt.show()

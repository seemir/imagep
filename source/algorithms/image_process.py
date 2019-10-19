# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from abc import ABC, abstractmethod
from skimage import color
from warnings import warn
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

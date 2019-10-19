# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from source.algorithms import Equalization
from source.algorithms import Otsu
import matplotlib.image as mpimg

image = mpimg.imread('source/images/bees.jpg')
otsu_image = Otsu(image)
otsu_image.compare_images()

equal_image = Equalization(image)
equal_image.compare_images()

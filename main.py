# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from source.algorithms.otsu import Otsu
from matplotlib.image import imread

image = imread('source/images/bees.jpg')
otsu_image = Otsu(image)
otsu_image.compare_images()

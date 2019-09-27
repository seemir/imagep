# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from source.algorithms.otsu import Otsu
from skimage import io

bees = io.imread('source/images/bees.jpg')
otsu_image = Otsu(bees)
otsu_image.compare_images()

# -*- coding: utf-8 -*-

__author__ = 'Samir Adrik'
__email__ = 'samir.adrik@gmail.com'

from source.algorithms.otsu import Otsu
from skimage import io

bee_image = io.imread('source/images/lenna.png')
otsu_image = Otsu(bee_image)
otsu_image.compare_images()

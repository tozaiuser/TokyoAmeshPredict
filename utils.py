#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Util functions"""

import numpy as np
import os
import glob
from skimage import io
from settings import Settings

class ImageBase(object):
    u'''Image base class for loading image'''

    def __init__(self):
        u'''Init function'''
        self.filepath = None
        self.size = None
        self.values = None

    def load_data(self, filepath):
        u'''Load data from filepath'''
        if os.path.exists(filepath):
            dt = Settings.DATA
            dt_max = Settings.DATA_MAX

            self.filepath = filepath
            img = io.imread(filepath)
            (nrow, ncol) = (img.shape[0], img.shape[1])
            self.size = (nrow, ncol) 
            # Initialize values for image
            self.values = np.zeros((nrow, ncol), dtype=np.float32)
            # Convert RBG 3 channels to float32 value in range [0, 1]
            for i in range(nrow):
                for j in range(ncol):
                    tpr = (img[i, j, 0], img[i, j, 1], img[i, j, 2])
                    if tpr in dt:
                        self.values[i, j] = dt[tpr] / dt_max
                    else:
                        self.values[i, j] = 0

    def get_data_pattern(self, filepath):
        u'''Get patterns of data'''
        if os.path.exists(filepath):
            self.filepath = filepath
            img = io.imread(filepath)
            (nrow, ncol) = (img.shape[0], img.shape[1])
            sample = set()
            for i in range(nrow):
                for j in range(ncol):
                    #tmp = np.int32(img[i, j, 0] << 8) | np.int32(img[i, j, 1])
                    #tmp = np.int32(tmp << 8) | np.int32(img[i, j, 2])
                    sample.add((img[i, j, 0], img[i, j, 1], img[i, j, 2]))
            return sample

class Flows(object):
    u'''Flows vector for each pixel'''

    def __init__(self, size):
        u'''Init function'''
        self.size = size
        self.values = np.zeros(size, dtype=[('var1','f4'),('var2','f4')])

class Grid(object):
    u'''Control grids'''

    def __init__(self, nsize, spacing):
        u'''Init function'''
        self.origin = (-2*spacing[0], -2*spacing[1])
        srow = int( (nsize[0] - 1 - self.origin[0]) / spacing[0] ) + 2
        scol = int( (nsize[1] - 1 - self.origin[1]) / spacing[1] ) + 2
        self.size = (srow, scol)
        self.spacing = spacing
        self.values = np.zeros((srow, scol), dtype=[('var1','f4'),('var2','f4')])



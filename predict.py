#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
from datetime import datetime, date, time, timedelta
from matplotlib import pyplot as plt
from skimage import io, color

DATA_DIR = r'../TOKYO_AMESH_IMAGE/000'

def transform(bf_img, flow):
    u'''Linear interpolator for transform'''
    # Check size of bf_img and af_img
    assert(bf_img.shape[0:2] == flow.shape[0:2])
    assert(flow.shape[2] == 2)

    s = bf_img.shape
    if len(s) != 3:
        prev_img = np.zeros(s[0], s[1], 1, dtype=bf_img.dtype)
        prev_img[:, :, 0] = bf_img    
    else:
        prev_img = bf_img

    (ysize, xsize, csize) = prev_img.shape
    af_img = np.zeros_like(prev_img)
    xmax = xsize - 1
    ymax = ysize - 1
    for y in range(ysize):
        for x in range(xsize):
            for c in range(csize):
                xnew = x - flow[y, x, 0]
                ynew = y - flow[y, x, 1]
                g = 0
                if xnew >= 0 and xnew < xsize and \
                    ynew >= 0 and ynew < ysize:
                    # Linear interpolator inside
                    px = int(xnew)
                    py = int(ynew)
                    dx = xnew - px
                    dy = ynew - py
                    px1 = min(px+1, xmax)
                    py1 = min(py+1, ymax)
                    g00 = prev_img[py , px , c]
                    g10 = prev_img[py1, px , c]
                    g01 = prev_img[py , px1, c]
                    g11 = prev_img[py1, px1, c]
                    g0  = dx * float(g01) + (1.0 - dx) * float(g00)
                    g1  = dx * float(g11) + (1.0 - dx) * float(g10)
                    g   = dy * g1 + (1 - dy) * g0
                else:
                    # Nearest interpolator outside
                    px = int(xnew)
                    px = max(px, 0)
                    px = min(px, xmax)
                    py = int(ynew)
                    py = max(py, 0)
                    py = min(py, ymax)
                    g = prev_img[py, px, c]
                af_img[y, x, c] = int(g)
    if len(s) != 3:
        return af_img[:, :, 0]
    else:
        return af_img

class Predict(object):
    u'''Predict class, load data and perform prediction'''
    
    def __init__(self, timestr, num_data):
        u'''Init function'''
        self.timestr = timestr
        self.num_data = num_data
        self.datapath = None
        self.before = None
        self.after = None
        self.current = None
        self.result = None
        self.optical = None

    def load_all_data_path(self):
        u'''Load previous images path before timestr'''
        year = int(self.timestr[0:4])
        month = int(self.timestr[4:6])
        day = int(self.timestr[6:8])
        hour = int(self.timestr[8:10])
        mins = int(self.timestr[10:12])

        d = date(year, month, day)
        t = time(hour, mins)
        dt = datetime.combine(d, t)
        # Time interval
        diff = timedelta(minutes=10)

        # Get the paths of images, load for data
        show_files = []
        for i in range(self.num_data):
            dt = dt - diff
            filename = str(dt.year).zfill(4) + str(dt.month).zfill(2) + \
                str(dt.day).zfill(2) + str(dt.hour).zfill(2) + str(dt.minute).zfill(2) + '.gif'
            folder_dir = os.path.join(DATA_DIR, str(dt.year))
            filepath = os.path.join(folder_dir, filename)
            if os.path.exists(filepath):
                show_files.append(filepath)

        self.datapath = show_files

        # Load for current image (to compare with predicted image)
        folder_dir = os.path.join(DATA_DIR, str(year))
        filename = self.timestr + '.gif'
        filepath = os.path.join(folder_dir, filename)
        if os.path.exists(filepath):
            self.current = io.imread(filepath)

    def run_simple(self):
        u'''Simple prediction'''
        if len(self.datapath) >= 2:
            # Use only two previous images
            af_img = io.imread(self.datapath[0])
            bf_img = io.imread(self.datapath[1])
            
            #af_img = io.imread(r'./penguin1.png')
            #bf_img = io.imread(r'./penguin1b.png')

            # Convert to gray image
            af_gray = color.rgb2gray(af_img)
            bf_gray = color.rgb2gray(bf_img)

            # Calculate density flow
            # Small -> WHY?
            flow = cv2.calcOpticalFlowFarneback(bf_gray, af_gray, \
                0.5, 6, 15, 10, 5, 1.2, 0)
            print flow[:, :, 0].min(), flow[:, :, 1].max()  
            self.before = bf_img
            self.after = af_img
            #self.result = self.current
            self.result = transform(af_img, flow)
            
            # Color code the result for better visualization of optical flow. 
            # Direction corresponds to Hue value of the image. 
            # Magnitude corresponds to Value plane
            
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros_like(af_img)
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            self.optical = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)        

    def run(self):
        u'''This one is for complicated predictor'''

    def display_compare(self):
        u'''Display predicted result and real image with difference'''
        row, col = 3, 2
        fig, axes = plt.subplots(row, col)
        for i in range(row):
            for j in range(col):
                plt.setp(axes[i, j].get_xticklines(), visible=False)
                plt.setp(axes[i, j].get_xticklabels(), visible=False)

                plt.setp(axes[i, j].get_yticklines(), visible=False)
                plt.setp(axes[i, j].get_yticklabels(), visible=False)

        # Load images
        axes[0, 0].imshow(self.optical, cmap='gray')
        axes[0, 0].set_title("Optical flow")

        axes[1, 0].imshow(self.after, cmap='gray')
        axes[1, 0].set_title("Prev1")

        axes[2, 0].imshow(self.before, cmap='gray')
        axes[2, 0].set_title("Prev2")

        axes[0, 1].imshow(self.after, cmap='gray')
        axes[0, 1].set_title("Prev 1")
        
        axes[1, 1].imshow(self.result, cmap='gray')
        axes[1, 1].set_title("Predicted")

        axes[2, 1].imshow(self.current, cmap='gray')
        axes[2, 1].set_title("Real")

        plt.show()

def _test():
    predictor = Predict('200905290240', 2)
    predictor.load_all_data_path()
    predictor.run_simple()
    predictor.display_compare()
    #print predictor.result.astype(np.float32) - predictor.after.astype(np.float32)
    
if __name__ == '__main__':
    _test()
    




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regist params namespace"""

from optimizer import RegularOptimizer
from utils import Grid

def CalMAD(moving, fixed):
	# NEED TO IMPLEMENTATION
	pass

def CalMADByGrid(moving, fixed, vgrid):
	# NEED TO IMPLEMENTATION
	        u'''calculate Mean Absolute Difference'''
        assert(moving.shape[0:1] == fixed.shape[0:1])

        (ysize,xsize) = moving.shape

        xmax = xsize - 1
        ymax = ysize - 1
        Ed = 0

        for y in range(ysize):
            for x in range(xsize):
                X = (x // 4) + 1
                Y = (y // 4) + 1
                rx = x % 4
                ry = y % 4
                flowx = ((4 - rx) * (4 - ry) * vgrid[Y,X,0] + (4 - rx) * ry * vgrid[Y+1,X,0] + rx * (4 - ry) * vgrid[Y,X+1,0] + rx * ry * vgrid[Y+1,X+1,0]) / 16.0
                flowy = ((4 - rx) * (4 - ry) * vgrid[Y,X,1] + (4 - rx) * ry * vgrid[Y+1,X,1] + rx * (4 - ry) * vgrid[Y,X+1,1] + rx * ry * vgrid[Y+1,X+1,1]) / 16.0
                xnew = x + flowx
                ynew = y + flowy
                g = 0.0

                if xnew >= 0 and xnew < xsize and \
                    ynew >= 0 and ynew < ysize:
                    # Linear interpolator inside
                    px = int(xnew)
                    py = int(ynew)
                    dx = xnew - px
                    dy = ynew - py
                    px1 = min(px+1, xmax)
                    py1 = min(py+1, ymax)
                    g00 = moving[py , px]
                    g10 = moving[py1, px]
                    g01 = moving[py , px1]
                    g11 = moving[py1, px1]
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
                    g = moving[py, px]
                
                Ed += fabs(fixed[y,x] - g)
        Ed /= (xmax * ymax)
        return Ed


def CalLinearDeriv(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	(Ysize, Xsize, Z) = vgrid.shape
        for Y in range(Ysize):
            y = 4 * Y 
            for X in range(Xsize):
                x = 4 * X
                deri[Y,X,0] = 0
                deri[Y,X,1] = 0
                for i in range(-3,4):
                    if i < 0:
                        flagi = -1
                    else:
                        flagi = 1
                    for k in range(-3,4):
                        if k < 0:
                            flagk = -1
                        else:
                            flagk = 1

                        rx = fabs(k)
                        ry = fabs(i)
                        flowx = ((4 - rx) * (4 - ry) * vgrid[Y,X,0] + (4 - rx) * ry * vgrid[Y+flagk,X,0] + rx * (4 - ry) * vgrid[Y,X+flagi,0] + rx * ry * vgrid[Y+flagk,X+flagi,0]) / 16.0
                        flowy = ((4 - rx) * (4 - ry) * vgrid[Y,X,1] + (4 - rx) * ry * vgrid[Y+flagi,X,1] + rx * (4 - ry) * vgrid[Y,X+flagk,1] + rx * ry * vgrid[Y+flagi,X+flagk,1]) / 16.0
                        xnew = x + flowx
                        ynew = y + flowy
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
                            g00 = moving[py , px ]
                            g10 = moving[py1, px ]
                            g01 = moving[py , px1]
                            g11 = moving[py1, px1]
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
                            g = moving[py, px, c]

                        if fixed[y+i,x+k] - g > 0:
                            flag = -1
                        else:
                            flag = 1
                        deri[Y,X,0] += flag * ((4 - ry) * (4 - rx) / 16) * vgrad[y+i,x+k,0] 
                        deri[Y,X,1] += flag * ((4 - ry) * (4 - rx) / 16) * vgrad[y+i,x+k,1] 
                        
        return deri

def CalCubicDeriv(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	pass

def CalLinearDerivAndMAD(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	pass

def CalCubicDerivAndMAD(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	pass

class RegularRegist(object):
	u'''Registration class'''

	def __init__(self, moving, fixed):
		u'''Init regist class with two images'''
		assert(moving.size == fixed.size)

		self.moving = moving
		self.fixed = fixed
		self.movgrad = None
		self.moved = None
		self.vgrid = Grid(moving.size, (4, 4))
		# Initialize optimizer with vgrid and dervgrid
		self.optimizer = RegularOptimizer(self.vgrid, Grid(moving.size, (4, 4)))

	def run(self):
		u'''Run to find vgrid'''
		self.optimizer.vdervgrid = CalLinearDeriv(self.moving, self.fixed, self.movgrad, \
			self.optimizer.vgrid)

		while self.optimizer.forward():
			self.optimizer.vdervgrid = CalLinearDeriv(self.moving, self.fixed, self.movgrad, \
				self.optimizer.vgrid)
			self.optimizer.cost = CalMADByGrid(self.moving, self.fixed, \
				self.optimizer.vgrid)

			# Or can call
			#self.optimizer.vdervgrid, self.optimizer.cost = \
			#	CalLinearDerivAndMAD(self.moving, self.fixed, self.movgrad, \
			#		self.optimizer.vgrid)

		self.vgrid = self.optimizer.vgrid

	def linear_transform(self):
		u'''Linear transform moving image by vgrid'''
		# NEED TO IMPLEMENTATION
		pass

	def cubic_transform(self):
		u'''Linear transform moving image by vgrid'''
		# NEED TO IMPLEMENTATION
		pass

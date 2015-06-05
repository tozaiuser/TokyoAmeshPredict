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
                X = (x // spacing[0]) + 1
                Y = (y // spacing[1]) + 1
                rx = (x % spacing[0]) / spacing[0]
                ry = (y % spacing[1]) / spacing[1]
		vx1 = vgrid[Y, X, 0]
		vx2 = vgrid[Y, X+1, 0]
		vx3 = vgrid[Y+1, X, 0]
		vx4 = vgrid[Y+1, X+1, 0]
		vy1 = vgrid[Y, X, 1]
		vy2 = vgrid[Y, X+1, 1]
		vy3 = vgrid[Y+1, X, 1]
		vy4 = vgrid[Y+1, X+1, 1] 
                flowx1 = (1 - rx) * vx1 + rx * vx2 
                flowx2 = (1 - rx) * vx3 + rx * vx4
		flowx = (1 - ry) * flowx1 + ry * flowx2 
                flowy1 = (1 - rx) * vy1 + rx * vy2 
                flowy2 = (1 - rx) * vy3 + rx * vy4
		flowy = (1 - ry) * flowy1 + ry * flowy2
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
        Ed /= (xsize * ysize)
        return Ed


def CalLinearDeriv(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	(Ysize, Xsize, Z) = vgrid.shape
        for Y in range(Ysize):
            y = spacing[1] * Y 
            for X in range(Xsize):
                x = spacing[0] * X
                deri[Y,X,0] = 0
                deri[Y,X,1] = 0
                for i in range(1:spacing[1]):
                    for k in range(1:spacing[0]):
                        rx = k / spacing[0]
                        ry = i / spacing[1]
			vx1 = vgrid[Y, X, 0]
			vx2 = vgrid[Y, X+1, 0]
			vx3 = vgrid[Y+1, X, 0]
			vx4 = vgrid[Y+1, X+1, 0]
			vy1 = vgrid[Y, X, 1]
			vy2 = vgrid[Y, X+1, 1]
			vy3 = vgrid[Y+1, X, 1]
			vy4 = vgrid[Y+1, X+1, 1] 
			flowx1 = (1 - rx) * vx1 + rx * vx2 
			flowx2 = (1 - rx) * vx3 + rx * vx4
			flowx = (1 - ry) * flowx1 + ry * flowx2 
			flowy1 = (1 - rx) * vy1 + rx * vy2 
			flowy2 = (1 - rx) * vy3 + rx * vy4
			flowy = (1 - ry) * flowy1 + ry * flowy2
                        if i == 1:
				#vertical & horizontal line
				flowverx = (1 - ry) * vx1 + ry * vx3
				flowvery = (1 - ry) * vy1 + ry * vy3
				flowhorx = (1 - rx) * vx1 + rx * vx2
				flowhory = (1 - rx) * vy1 + rx * vy2 
				xvernew = x + flowverx
				yvernew = y + k + flowvery
				xhornew = x + k + flowhorx
				yhornew = y + flowhory
				if xvernew >= 0 and xvernew < xsize and \
                                yvernew >= 0 and yvernew < ysize:
					# Linear interpolator inside
					px = int(xvernew)
					py = int(yvernew)
					dx = xvernew - px
					dy = yvernew - py
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
					px = int(xvernew)
					px = max(px, 0)
					px = min(px, xmax)
					py = int(yvernew)
					py = max(py, 0)
					py = min(py, ymax)
					g = moving[py, px]
				if fixed[y+i,x] - g > 0:
					flagver = -1
				else:
					flagver = 1
				if xhornew >= 0 and xhornew < xsize and \
                                yhornew >= 0 and yhornew < ysize:
					# Linear interpolator inside
					px = int(xhornew)
					py = int(yhornew)
					dx = xhornew - px
					dy = yhornew - py
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
					px = int(xhornew)
					px = max(px, 0)
					px = min(px, xmax)
					py = int(yhornew)
					py = max(py, 0)
					py = min(py, ymax)
					g = moving[py, px]
				if fixed[y,x+i] - g > 0:
					flaghor = -1
				else:
					flaghor = 1
				deriver[Y,X,0] += flagver * (1 - ry) * vgrad[y+k,x,0] 
				deriver[Y,X,1] += flagver * (1 - ry) * vgrad[y+k,x,1]
				derihor[Y,X,0] += flaghor * (1 - rx) * vgrad[y,x+k,0] 
				derihor[Y,X,1] += flagver * (1 - rx) * vgrad[y,x+k,1] 

                        xnew = x + k + flowx
                        ynew = y + i + flowy
                        g = 0
			#inside a grid
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
                            g = moving[py, px]

                        if fixed[y+i,x+k] - g > 0:
                            flag = -1
                        else:
                            flag = 1
                        derisq[Y,X,0] += flag * (1 - ry) * (1 - rx) * vgrad[y+i,x+k,0] 
                        derisq[Y,X,1] += flag * (1 - ry) * (1 - rx) * vgrad[y+i,x+k,1] 
	for Y in range (Ysize):
		for X in range (Xsize):
			x = spacing[0] * X
			y = spacing[1] * Y
			xnew = x + vgrid[Y,X,0]
			ynew = y + vgrid[Y,X,1]
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
                            g   = dy * g1 + (1.0 - dy) * g0
			    
                        else:
                            # Nearest interpolator outside
                            px = int(xnew)
                            px = max(px, 0)
                            px = min(px, xmax)
                            py = int(ynew)
                            py = max(py, 0)
                            py = min(py, ymax)
                            g = moving[py, px]
                        if fixed[y+i,x+k] - g > 0:
                            flag = -1
                        else:
                            flag = 1
			#その点の計算に、上下左右の線上とグリッド内の点の和を加える
			deri[Y,X,0] = flag * vgrad[y,x,0] + derisq[Y,X,0] + derisq[Y-1,X,0] + derisq[Y,X-1,0] + derisq[Y-1,X-1,0] + deriver[Y,X,0] + deriver[Y-1,X,0] + derihor[Y,X,0] + derihor[Y,X-1,0]
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

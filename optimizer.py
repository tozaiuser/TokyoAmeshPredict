#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimizer class"""

class RegularOptimizer(object):
	u'''Regular gradient descent optimizer'''

	def __init__(self, vgrid, dervgrid, max_iters=100, base_lr=0.01):
		u'''Initialize regular optimizer'''
		# max_iters = max iterations
		# base_lr : the first learning rate
		self.iter = 0
		self.max_iters = max_iters
		self.lr = base_lr
		self.vgrid = vgrid
		self.vdervgrid = dervgrid
		self.cost = None

	def converged(self):
		u'''Check convergence condition'''
		# NEED TO IMPLEMENTATION 
		if self.iter >= self.max_iters:
			return True
		else:
			return False

	def forward(self):
		u'''Forward the optimizer to the next step'''
		# Check converged condition
		if self.converged():
			return False
		else:
			# Update vgrid by vdervgrid
			(g_row, g_col) = self.vgrid.size
			for i in range(g_row):
				for j in range(g_col):
					# NEED TO IMPLEMENTATION
					# self.vgrid[i, j] = ???
					pass

			# Increase interation step
			print self.iter
			self.iter += 1
			return True
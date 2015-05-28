#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimizer class"""

from math import sqrt


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
		self.dervgrid = dervgrid
		self.cost = None

	def converged(self):
		u'''Check convergence condition'''
		(g_row, g_col) = self.vgrid.size
		for i in range(g_row):
			for j in range(g_col):
				diff0 = self.lr * self.dervgrid.values[i, j][0]
				diff1 = self.lr * self.dervgrid.values[i, j][1]
				val0 = self.vgrid.values[i, j][0]
				val1 = self.vgrid.values[i, j][1]
				if sqrt(diff0 ** 2 + diff1 ** 2) < 0.001 * sqrt(val0 ** 2 + val1 ** 2):
					return True
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
			# Update vgrid by dervgrid
			(g_row, g_col) = self.vgrid.size
			for i in range(g_row):
				for j in range(g_col):
					diff0 = self.lr * self.dervgrid.values[i, j][0]
					diff1 = self.lr * self.dervgrid.values[i, j][1]
					self.vgrid.values[i, j][0] -= diff0
					self.vgrid.values[i, j][1] -= diff1
			# Increase interation step
			print self.iter
			self.iter += 1
			if self.iter % 10 == 0:
				self.lr *= 0.9
			return True

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
	pass

def CalLinearDeriv(moving, fixed, vgrad, vgrid):
	# NEED TO IMPLEMENTATION
	pass

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

	def transform(self):
		u'''Transform moving image by vgrid'''
		pass

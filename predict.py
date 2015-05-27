#!/usr/bin/env python
# -*- coding: utf-8 -*-
u'''Predict module'''

from utils import ImageBase, Flows, Grid
from settings import Settings
from regist import RegularRegist

def _test():
    moving = ImageBase()
    moving.load_data(r'../TOKYO_AMESH_IMAGE/000/2009/200905290240.gif')
    flows = Flows(moving.size)
    vgrid = Grid(moving.size, (4, 4))
    print vgrid.size

    fixed = ImageBase()
    fixed.load_data(r'../TOKYO_AMESH_IMAGE/000/2009/200905290250.gif')
    regist = RegularRegist(moving, fixed)
    regist.run()
    print regist.optimizer.vgrid.values
    print regist.fixed.values
    print regist.optimizer.cost

def test_values_img():
    u'''Test value in images'''
    sample = set()
    for year in range(2009, 2010):
        folder_dir = os.path.join(Settings.DATA_DIR, str(year))
        for filepath in glob.glob(folder_dir + '/*.gif'):
            imgb = ImageBase()
            sample = sample.union(imgb.get_data(filepath))
            print filepath, sample

if __name__ == '__main__':
    _test()
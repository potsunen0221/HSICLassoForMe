#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)

from future import standard_library

from pyHSICLasso import HSICLasso

import scipy.io as sio

standard_library.install_aliases()

from functools import reduce
import pandas as pd
import glob
import re
import argparse as arg
import numpy as np

import os
import sys

import shutil
import random

from sklearn.preprocessing import LabelEncoder

class myFeatureSelection:
    
    def __init__ (self, all_data_list, f_num, neighbor_num, B, learning):
        self.X, self.Y, self.TargetID = self.readData(all_data_list)
        self.feat_num = f_num
        self.neighbor_num = neighbor_num
        self.B = B
        self.learning = learning
    
    def read_data(self, fpath):
        m = re.match('.+/(.+?).txt', fpath)
        name = m.group(1)
        return pd.read_csv(fpath, sep='\t', skipinitialspace=True, header=0,
                           names=['TargetID', name],
                           dtype={'TargetID': object, name: 'float32'}, index_col='TargetID')

    def readData(self, all_data_list):
        l_list=pd.read_csv(all_data_list, sep='\t', header=0, names=['Target', 'Name', 'Group'])
        target = l_list.Target.values
    #     group = l_list.Group.values
        datas = [self.read_data(path) for path in l_list.Name]
        data = datas[0].join(datas[1:])
        data.fillna(0.5, inplace=True)
        print(len(data.columns)," samples x ",len(data), " features ")
        return data.T.values, target, data.T.columns
    
    def myHSICLasso(self, feat_num, neighbor_num, B, learning):
    
        if len(self.X) < B:
            print(f'B {B} must be smaller than the number of samples {len(self.X)}.')
            print(f'Equate B with the number of samples {len(self.X)}.')
            B = len(self.X)

        print(f'Selecting {feat_num} features by HSIC Lasso ', end='')

        hsic_lasso = HSICLasso()

        if learning == 'r':
            print('regression.')
            hsic_lasso.input(self.X,self.Y,featname=self.TargetID)
            hsic_lasso.regression(feat_num, max_neighbors=neighbor_num, B=B, n_jobs=-1)
        elif learning == 'c':
            print('classification.')
            le = LabelEncoder()
            y = le.fit_transform(self.Y)
            hsic_lasso.input(self.X,y,featname=self.TargetID)
            hsic_lasso.classification(feat_num, max_neighbors=neighbor_num, B=B, n_jobs=-1)

        hsic_lasso.dump()

        os.makedirs('results', exist_ok=True)

        hsic_lasso.plot_path(filepath='results/path.png')

        feature_index = hsic_lasso.get_features()
        feature_score = hsic_lasso.get_index_score()
        hsic_selected = pd.DataFrame({'TargetID':feature_index,'Score':feature_score})

        neighbor_index = np.array([flatten for inner in [hsic_lasso.get_features_neighbors(feat_index=i,num_neighbors=neighbor_num) for i in range(feat_num)] for flatten in inner])
        neighbor_score = np.array([flatten for inner in [hsic_lasso.get_index_neighbors_score(feat_index=i,num_neighbors=neighbor_num) for i in range(feat_num)] for flatten in inner])
        neighbor_runk = np.array([flatten for inner in [['Neighbor%d' % i for i in range(1,neighbor_num+1)] for j in range(feat_num)] for flatten in inner])
        feat_rep = np.array([flatten for inner in [[i for j in range(neighbor_num)] for i in feature_index] for flatten in inner])
        hsic_neighbor = pd.DataFrame({'Feature':feat_rep,'Neighbor_runk':neighbor_runk,'TargetID':neighbor_index,'Score':neighbor_score})

        return hsic_selected, hsic_neighbor
    
    def main(self):
        
        print('feat_num : ',self.feat_num)
        print('neighbor_num : ',self.neighbor_num)
        print('B : ',self.B)
        print('Learning : ',self.learning)
        
        hsic_selected, hsic_neighbor = self.myHSICLasso(self.feat_num, self.neighbor_num, self.B, self.learning)
        hsic_selected.to_csv('results/Selected_features.txt',sep='\t',index=False)
        hsic_neighbor.to_csv('results/Neighbors.txt',sep='\t',index=False)
        
if __name__ == "__main__":
  
    parser = arg.ArgumentParser()
    parser.add_argument('all_data_list', help='学習・テストで使用するデータ全てのファイルパスが記載されたファイル、ヘッダ行あり、タブ区切り')
    parser.add_argument('-f','--f_num', help='Lassoにより選択される特徴数. default=10', default=10, type=int)
    parser.add_argument('-n','--n_num', help='選択された特徴の近傍の数. default=10(MAX=10)', default=10, type=int)
    parser.add_argument('-b','--B', help='Block数. 変える必要あまりなし. default=0', default=0, type=int)
    parser.add_argument('-l','--learning_type', help="'r' : regression  or 'c' : classification. default='r'", default='r', type=str)
    args = parser.parse_args()
    
    print('list : ',args.all_data_list)
    print('f_num : ',args.f_num)
    print('n_num : ',args.n_num)
    print('B : ',args.B)
    print('learning : ',args.learning_type)
  
    s = myFeatureSelection(args.all_data_list, args.f_num, args.n_num, args.B, args.learning_type)
    s.main()

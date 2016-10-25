#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm, datasets

import os
import sys
import math
import re
from os import path
from scipy import spatial
import pickle

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from tfidf.dg_lr import *

if len(sys.argv) != 3:
    print "input:doc-folder,idf-file"
    sys.exit(1)




wtol = readwl("/home/ec2-user/git/statresult/wordslist_dsw.txt")
a = np.load('/home/ec2-user/data/classinfo/vt-kk.npy')#lsa result
fidf = open(sys.argv[2],'r')
idf = {}
for line in fidf:
    line = line.strip('\n')
    line = line.split()
    idf[line[0]] = float(line[1])
fidf.close()
        



kk = a.shape[0]
root = sys.argv[1]

X = []
y = []
for root, dirs, files in os.walk(sys.argv[1]):
    for name in files:
        filename = root + '/' + name
        if filename[len(filename)-1] == 't':
            fin = open(filename,'r')
            cll = set()
            temp = fin.read()
            fin.close()
            cl = re.search(r'(【国際特許分類第.*版】.*?)([A-H][0-9]+?[A-Z])',temp,re.DOTALL)
            if cl == None:
                continue
            fin = open(filename+'.fq')
            temp = fin.readlines()
            fin.close()
            X.append(vecof2(temp[1:],idf,a,wtol,kk))
            y.append(cl.group(2))
h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
fio = open('rbf_svc.txt','w')
pickle.dump(rbf_svc,fio)
fio.close()
del rbf_svc
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
fio = open('poly_svc.txt','w')
pickle.dump(poly_svc,fio)
fio.close()




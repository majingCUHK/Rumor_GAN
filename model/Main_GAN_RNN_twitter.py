# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of GAN-RNN: both Generater & discriminator are RNN (2 classes)
@author: majing
@module: sequence words input, GAN
@variable: Nepoch, lr_g, lr_d
@time: Jun 7, 2018
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os

from model_GAN_RNN import GAN
from train import *
from evaluate import *
from Util import *

import numpy as np
from numpy.testing import assert_array_almost_equal
import time
import datetime
import random

vocabulary_size = 5000
hidden_dim = 100
Nclass = 2
Nepoch = 300  # main epoch
Nepoch_G = 51 # pre-Train G
Nepoch_D = 121 # pre-Train D

lr_g = 0.005
lr_d = 0.005

obj = "twitter"#"test-" # dataset
fold = "0"

unit="GAN-RNN-"+obj+str(fold)
modelPath = "../param/param-"+unit+".npz" 

unit_dis="RNN-"+obj+str(fold)
modelPath_dis = "../param/param-"+unit_dis+".npz" 

unit_pre="GAN-RNN-pre-"+obj+str(fold)
modelPath_pre = "../param/param-"+unit_pre+".npz" 

trainPath = "../nfold/TrainSet_"+obj+fold+".txt" 
testPath = "../nfold/TestSet_"+obj+fold+".txt"
labelPath = "../resource/"+obj+"_labels.txt"
textPath = '../resource/'+obj+'.vol_5000.txt' 
#textPath = '../resource/'+obj+'_new.words.ABS.vol_5000_intv_40.txt' 

################################### tools #####################################
def dic2matrix(dicW):
    # form: dicW = {ts:[index:wordfreq]}
    X = []
    keyset = dicW.keys()
    timestamps = sorted(keyset)
    for ts in timestamps:
        x = [0 for i in range(vocabulary_size)]
        for pair in dicW[ts]:
            x[ int(pair.split(':')[0]) ] = int(pair.split(':')[1])
        X.append( x )
    return X

def dic2matrixNorm(dicW): 
    # form: dicW = {ts:[index:wordfreq]}
    X = []
    keyset = dicW.keys()
    timestamps = sorted(keyset)
    for ts in timestamps:
        x = [0 for i in range(vocabulary_size)]
        Sum, Min, Max = 0, 0.0, 0.0
        for pair in dicW[ts]:
            Sum += int(pair.split(':')[1])
            if int(pair.split(':')[1]) > Max:
               Max = int(pair.split(':')[1])
        for pair in dicW[ts]:       
            x[ int(pair.split(':')[0]) ] = round( (float(pair.split(':')[1])-Min) / (Max-Min), 4)
        X.append( x )
    return X
    
labelset_true = ['true', 'non-rumour']
labelset_false = ['false', 'rumour']
def loadLabel(label):
    #print label
    #y_train, y_train_gen = [], []
    if label in labelset_true:
       y_train = [1,0]
       y_train_gen = [0,1]
    if label in labelset_false:
       y_train = [0,1] 
       y_train_gen = [1,0]
    return y_train, y_train_gen
              
################################# loas data ###################################
def loadData():
    print "loading labels",
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid, label = line.split('\t')[0], line.split('\t')[1].lower()
        labelDic[eid] = label
    print len(labelDic)
    
    print "reading events", ## X
    textDic = {}
    for line in open(textPath):
        line = line.rstrip()
        if len(line.split('\t')) < 3: continue
        eid, ts, Vec = line.split('\t')[0], int(line.split('\t')[1]), line.split('\t')[2].split(' ')
        if textDic.has_key(eid):
           textDic[eid][ts] = Vec
        else:
           textDic[eid] = {ts: Vec} 
    print len(textDic)
    
    print "loading train set", 
    x_word_train, y_train, y_gen_train, Len_train, c = [], [], [], [], 0
    index_true, index_false = [], []
    for eid in open(trainPath):
        #if c > 8: break
        eid = eid.rstrip()
        if not labelDic.has_key(eid): continue
        if not textDic.has_key(eid): continue 
        ## 1. load label
        label = labelDic[eid]
        if label in labelset_true: index_true.append(c)
        if label in labelset_false: index_false.append(c)

        y, y_gen = loadLabel(label)
        y_train.append(y)
        y_gen_train.append(y_gen)
        Len_train.append( len(textDic[eid]) )
        #wordFreq = dic2matrix( textDic[eid] )
        wordFreq = dic2matrixNorm( textDic[eid] )
        x_word_train.append( wordFreq )
        c += 1
    print c
    
    print "loading test set", 
    x_word_test, y_test, Len_test, c = [], [], [], 0
    for eid in open(testPath):
        #if c > 4: break
        eid = eid.rstrip()
        if not labelDic.has_key(eid): continue
        if not textDic.has_key(eid): continue 
        ## 1. load label        
        label = labelDic[eid]
        y, y_gen = loadLabel(label)
        y_test.append(y)
        Len_test.append( len(textDic[eid]) )
        #wordFreq = dic2matrix( textDic[eid] )
        wordFreq = dic2matrixNorm( textDic[eid] )
        x_word_test.append( wordFreq )
        c += 1
    print c
    #print "train no:", len(x_word_train), len(y_train), len(y_gen_train), len(Len_train)
    #print "train no:", len(x_word_test), len(y_test), len(Len_test)    
    #print "dim1 for 0:", len(x_word_train[0]), len(x_word_train[0][0]), y_train[0], y_gen_train[0], Len_train[0]

    return x_word_train, y_train, y_gen_train, Len_train, x_word_test, y_test, index_true, index_false

##################################### MAIN ####################################        
## 1. load tree & word & index & label
x_word_train, y_train, yg_train, Len_train, x_word_test, y_test, index_true, index_false = loadData()

## 2. ini RNN model
t0 = time.time()
GANmodel = GAN(vocabulary_size, hidden_dim, Nclass)
t1 = time.time()
print 'GAN model established,', (t1-t0)/60

## 3. pre-train or load model
if os.path.isfile(modelPath):
   GANmodel = load_model(modelPath, GANmodel)   
   lr_d, lr_g = 0.0001, 0.0001
else:
   # pre train classifier
   if os.path.isfile(modelPath_dis):
      GANmodel = load_model_dis(modelPath_dis, GANmodel)
      #lr_d = 0.005
   else:
      pre_train_Discriminator(GANmodel, x_word_train, y_train, x_word_test, y_test, lr_d, Nepoch_D, modelPath_dis)
   #exit(0) 
   # pre train generator
   if os.path.isfile(modelPath_pre):
      GANmodel = load_model(modelPath_pre, GANmodel)
      #pre_train_Generator('rn', GANmodel, x_word_train, x_index_train, index_false, Len_train, y_train, yg_train, lr_g, Nepoch_G, modelPath_pre, floss)

      #lr_g = 0.005
   else:   
      pre_train_Generator('nr', GANmodel, x_word_train, index_true, Len_train, y_train, yg_train, lr_g, Nepoch_G, modelPath_pre)
      pre_train_Generator('rn', GANmodel, x_word_train, index_false, Len_train, y_train, yg_train, lr_g, Nepoch_G, modelPath_pre)
      #lr_g = 0.0001'''
      
train_Gen_Dis(GANmodel, x_word_train,Len_train, y_train, yg_train, index_true, index_false, x_word_test, y_test, lr_g, lr_d, Nepoch, modelPath)
     
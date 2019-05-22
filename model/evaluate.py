# -*- coding: utf-8 -*-
"""
@object: weibo & twitter
@task: split train & test, evaluate performance 
@author: majing
@variable: T, 
@time: Tue Nov 10 16:29:42 2015
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import random
import os
import re
import math
import numpy as np

################## evaluation of model result #####################

def evaluation_2class(prediction, y): # 4 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    e, RMSE, RMSE1, RMSE2 = 0.000001, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i][0])
        ##RMSE
        for j in range(len(y_i)):
            RMSE += (y_i[j]-p_i[j])**2
        RMSE1 += (y_i[0]-p_i[0])**2 
        RMSE2 += (y_i[1]-p_i[1])**2 
        ## Pre, Recall, F    
        Act = str(y_i.index(max(y_i))+1)  
        Pre = str(p_i.index(max(p_i))+1)
        
        ## for class 1
        if Act == '1' and Pre == '1': TP1 += 1
        if Act == '1' and Pre != '1': FN1 += 1    
        if Act != '1' and Pre == '1': FP1 += 1
        if Act != '1' and Pre != '1': TN1 += 1
        ## for class 2
        if Act == '2' and Pre == '2': TP2 += 1
        if Act == '2' and Pre != '2': FN2 += 1    
        if Act != '2' and Pre == '2': FP2 += 1
        if Act != '2' and Pre != '2': TN2 += 1    

    ## print result
    Acc_all = round( float(TP1+TP2)/float(len(y)+e), 4 )
    Prec1 = round( float(TP1)/float(TP1+FP1+e), 4 )
    Recll1 = round( float(TP1)/float(TP1+FN1+e), 4 )
    F1 = round( 2*Prec1*Recll1/(Prec1+Recll1+e), 4 )
    
    Prec2 = round( float(TP2)/float(TP2+FP2+e), 4 )
    Recll2 = round( float(TP2)/float(TP2+FN2+e), 4 )
    F2 = round( 2*Prec2*Recll2/(Prec2+Recll2+e), 4 )
    
    RMSE_all = round( ( RMSE/len(y) )**0.5, 4)
    RMSE_all_1 = round( ( RMSE1/len(y) )**0.5, 4)
    RMSE_all_2 = round( ( RMSE2/len(y) )**0.5, 4)

    RMSE_all_avg = round( ( RMSE_all_1+RMSE_all_2 )/2, 4)
    return ['acc:', Acc_all, 'nPrec:', Prec1, 'nRec:', Recll1, 'nF1:', F1,
                             'rPrec:', Prec2, 'rRec:', Recll2, 'rF1:', F2]
            
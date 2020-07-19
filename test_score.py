# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 06:14:19 2020

@author: wuxia
"""

import sys
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def evaluate(path_real, path_predict):
    real = []
    predict = []

    file = open(path_real)
    for row in file.readlines():
        clean = row.strip('\n')
        real.append(int(clean))
    file.close()

    file = open(path_predict)
    for row in file.readlines():
        clean = row.strip('\n')
        predict.append(int(clean))
    file.close()

    if len(predict)!=len(real): sys.exit('ERROR: O número de arquivos é diferente.')

    print("F1_macro.........: %.3f" %(f1_score(real, predict, average="macro") * 100))
    print("F1_micro.........: %.3f" %(f1_score(real, predict, average="micro") * 100))
    print("Precision..: %.3f" %(precision_score(real, predict, average="micro") * 100))
    print("Recall.....: %.3f" %(recall_score(real, predict, average="micro") * 100))
    print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))
 
path_real="F:/mlp/cw4/4k_golden.txt"
path_predict="F:/mlp/cw4/output.txt"
evaluate(path_real, path_predict)
'''
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

real= [1,1,1,1,1,1,1,1,1,1]
predict=[1,0,1,0,1,0,1,0,1,0]

print("==========WEIGHTED")

print("F1.........: %.3f" %(f1_score(real, predict, average="weighted") * 100))
print("Precision..: %.3f" %(precision_score(real, predict, average="weighted") * 100))
print("Recall.....: %.3f" %(recall_score(real, predict, average="weighted") * 100))
print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))

print("====================macro")

print("F1.........: %.3f" %(f1_score(real, predict, average="macro") * 100))
print("Precision..: %.3f" %(precision_score(real, predict, average="macro") * 100))
print("Recall.....: %.3f" %(recall_score(real, predict, average="macro") * 100))
print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))

print("========micro")

print("F1.........: %.3f" %(f1_score(real, predict, average="micro") * 100))
print("Precision..: %.3f" %(precision_score(real, predict, average="micro") * 100))
print("Recall.....: %.3f" %(recall_score(real, predict, average="micro") * 100))
print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))

print("========micro_macro")

print("F1.........: %.3f" %(f1_score(real, predict, average="macro") * 100))
print("Precision..: %.3f" %(precision_score(real,predict,average="micro") * 100))
print("Recall.....: %.3f" %(recall_score(real, predict,average="micro") * 100))
print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))
'''
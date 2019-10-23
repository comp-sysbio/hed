# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:00:40 2019

@author: yang_
"""
import numpy as np
import os
from scipy import interp
import scipy.stats
from sklearn import svm   
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

    
         
            

def test_svm_allresults(train_vector,train_label,testing_vector,testing_label,clf_time_bestparameters):
    
    t=len(testing_vector)#多少次实验
    testing_F1_score=[]
    test_svm_prediction_proba={}
    test_svm_pre_label={}
    test_svm_probability={}

    for i in range(1,t+1):
        times=str(i)
        c=clf_time_bestparameters[times][0]
        g=clf_time_bestparameters[times][1]
        test_vec=testing_vector[times]
        test_label=testing_label[times]
        x_train=train_vector[times]
        x_label=train_label[times]
        
        
        for i in range(len(test_label)):
            test_label[i]=int(test_label[i])
        
        model=svm.SVC(C=c, kernel='rbf', gamma=g,probability=True)
        model.fit(x_train,x_label)
        y_hat = model.predict(test_vec)
        test_svm_pre_label[times]=y_hat
        test_score=model.decision_function(test_vec)
        test_probability=model.predict_proba(test_vec)
        test_svm_prediction_proba[times]=test_score
        test_svm_probability[times]=test_probability
        
        testing_f_score=f1_score(test_label, y_hat)
        testing_F1_score.append(testing_f_score)

        
            
            
    print(testing_F1_score)
    return test_svm_prediction_proba,test_svm_pre_label,test_svm_probability


def get_results(test_svm_prediction_proba,test_svm_pre_label,test_svm_probability,testing_label,testing_katz_score,testing_katz_label,testing_rwrh_score,testing_rwrh_label):
    
       

    for i in range(1,T+1):
        times=str(i)
        
        
        
        pre= precision_score(testing_label[times], test_svm_pre_label[times])
        rec=recall_score(testing_label[times], test_svm_pre_label[times])
        f_score=f1_score(testing_label[times], test_svm_pre_label[times])
        precision.append(pre)
        recall.append(rec)
        f1_score_metapath.append(f_score)
        
        
       
        
        katz_time_score=testing_katz_score[times]
        katz_time_label=testing_katz_label[times]
        time_length=len(katz_time_score)
        label_length=time_length//2
        katz_pre_label=[0]*time_length
        katz_score_sortindex=sorted(range(time_length), key=lambda k: katz_time_score[k],reverse=True)
        for i in range(label_length):
            katz_pre_label[katz_score_sortindex[i]]=1
        pre_katz=precision_score(katz_time_label,katz_pre_label)
        rec_katz=recall_score(katz_time_label,katz_pre_label)
        f_score_katz=f1_score(katz_time_label,katz_pre_label)
        precision_katz.append(pre_katz)
        recall_katz.append(rec_katz)
        f1_score_katz.append(f_score_katz)
        
        rwrh_time_score=testing_rwrh_score[times]
        rwrh_time_label=testing_rwrh_label[times]
        time_length=len(rwrh_time_score)
        label_length=time_length//2
        rwrh_pre_label=[0]*time_length
        rwrh_score_sortindex=sorted(range(time_length), key=lambda k: rwrh_time_score[k],reverse=True)
        for i in range(label_length):
            rwrh_pre_label[rwrh_score_sortindex[i]]=1
        pre_rwrh=precision_score(rwrh_time_label,rwrh_pre_label)
        rec_rwrh=recall_score(rwrh_time_label,rwrh_pre_label)
        f_score_rwrh=f1_score(rwrh_time_label,rwrh_pre_label)
        precision_rwrh.append(pre_rwrh)
        recall_rwrh.append(rec_rwrh)
        f1_score_rwrh.append(f_score_rwrh)
        

        

precision=[]
recall=[]

precision_katz=[]
recall_katz=[]
precision_rwrh=[]
recall_rwrh=[]
f1_score_metapath=[]

f1_score_katz=[]
f1_score_rwrh=[]
test_svm_probability={}

T=10

for clf in ['1','2','3']:
    #test_svm_probability[clf]={}

    test_svm_prediction_proba,test_svm_pre_label,test_svm_probability[clf]=test_svm_allresults(training_vector[clf],training_label[clf],testing_vector[clf],testing_label[clf],clf_time_bestparameters[clf])
    get_results(test_svm_prediction_proba,test_svm_pre_label,test_svm_probability,testing_label[clf],testing_katz_score[clf],testing_katz_label[clf],testing_rwrh_score[clf],testing_rwrh_label[clf])

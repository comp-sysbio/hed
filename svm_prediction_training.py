# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:20:13 2019

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

from sklearn.model_selection import GridSearchCV   
 

drug_id={}
disease_id={}
id_drug={}
id_disease={}
training_vector={}
training_label={}
testing_vector={}
testing_label={}
clf_time_bestparameters={}
training_vector_plus={}
training_label_plus={}
testing_vector_plus={}
testing_label_plus={}
clf_time_bestparameters_plus={}

testing_list={}

testing_katz_score={}
testing_katz_label={}
testing_rwrh_score={}
testing_rwrh_label={}
c_g_can=[]
clf_c_g_max={}
test_pre_score={}
test_pre_label={}
test_pre_score_plus={}
test_pre_label_plus={}
T=10


def read_data(data_path):
    #data_file = open(data_path,'r')
    #node_vectors= open(r'..\Metapath\embeddings\CTD\drug_disease_embedding0_chuli.txt','r')
    guanlian_drug=open(data_path+'/'+'id_drug.txt','r')
    guanlian_disease=open(data_path+'/'+'id_disease.txt','r')
    for line in guanlian_drug:
        toks = line.strip().split('\t')
        drug_id[toks[1][1:]]=toks[0]
        id_drug[toks[0]]=toks[1][1:]
        
    guanlian_drug.close()
    for line in guanlian_disease:
        toks=line.strip().split('\t')
        disease_id[toks[1][1:]]=toks[0]
        id_disease[toks[0]]=toks[1][1:]
    guanlian_disease.close()


def get_training_vector_label(embedding_dir,clf_type,times):
    
    #drug_vector={}    
    #drug_plus_vector={}
    training_t_vector=[]
    training_t_label=[]
    testing_t_vector=[]
    testing_t_label=[]
    
    test_t_list=[]
    
    node_vectors= open(embedding_dir,'r')
    drug_vector={}
    disease_vector={}

    for line in node_vectors:
        toks=line.strip().split(' ')
        node_panduan=toks[0][0]
        node_name=toks[0][1:]
        vector=[]
        for i in range(len(toks)):
            if i>0:
                vector.append(toks[i])
        if node_panduan=='a':
            disease_vector[node_name]=vector
        if node_panduan=='v':
            drug_vector[node_name]=vector
            
    train_test_path='../data/CTD/metapath/right'+'/'+'%s'%clf_type+'/'+'%s'%times
    training_testing_times=os.listdir(train_test_path)
    for filename in training_testing_times:
        training_open=os.path.join(train_test_path,filename)
        file_open=open(training_open,'r')
        if filename == 'training.txt':
            #aaa=0
            for line in file_open:
                #print('line=',line)
                toks=line.strip().split('\t')
                drug=id_drug[toks[0]]
                disease=id_disease[toks[1]]
                label=toks[2]
                
#                if drug in drug_vector.keys() and disease in disease_vector.keys():
                #aaa=aaa+1
                drug_vec=drug_vector[drug]
                disease_vec=disease_vector[disease]
                rec=drug_vec+disease_vec
                training_t_vector.append(rec)
                training_t_label.append(label)
            #print ('train',aaa)
        if filename == 'testing.txt':
            #aaa=0
            for line in file_open:
                #print('line=',line)
                toks=line.strip().split('\t')
                test_t_list.append(toks)
                drug=id_drug[toks[0]]
                disease=id_disease[toks[1]]
                label=toks[2]
                
                if drug in drug_vector.keys() and disease in disease_vector.keys():
                #aaa=aaa+1
                    drug_vec=drug_vector[drug]
                    disease_vec=disease_vector[disease]
                    rec=drug_vec+disease_vec
                    testing_t_vector.append(rec)
                    testing_t_label.append(label)
            #print ('test',aaa)
  
    return  training_t_vector,training_t_label,testing_t_vector,testing_t_label,test_t_list
      
def show_accuracy(a, b):
    acc = a.ravel() == b.ravel()
    print ('accuracy %.2f%%' % (100*float(acc.sum()) / a.size))

def training_svm_parameter(X_train,y_train):

    best_c_g=[]
    for i in range(len(y_train)):
        y_train[i]=int(y_train[i])
    starttime = datetime.datetime.now()
    model = svm.SVC(kernel='rbf', probability=True)    
    param_test = {'C': list(np.logspace(-2, 3, 6)), 'gamma': list(np.logspace(-2, 3, 6))}    
    grid_search = GridSearchCV(estimator = model, param_grid=param_test,scoring='roc_auc',cv=10)    
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(X_train,y_train) 
    y_hat = model.predict(X_train)
    f_score=f1_score(y_train, y_hat)
    print(f_score)
    endtime = datetime.datetime.now()
    print ('running time =',(endtime - starttime).seconds)
    best_c=best_parameters['C']
    best_g=best_parameters['gamma']
    best_c_g.append(best_c)
    best_c_g.append(best_g)
    
    return best_c_g


def load_katz_rwrh_result():
    topk_path='../data/CTD/topk/right'
    topk_data=os.listdir(topk_path)
    result_katz={}
    clf_panduan=['1','2','3']

    result_katz={}
    result_rwrh={}

    for filename in topk_data:
        if filename in clf_panduan:

            result_katz[filename]={}
            result_rwrh[filename]={}
            topk_t=os.path.join(topk_path,filename)
            files_t=os.listdir(topk_t)
            for filename_t in files_t:
                
                times=filename_t.split('_')
                method=times[1].split('.')
    
                if method[0]=='katz':
                    print(method[0])
                    result_katz[filename][times[0]]={}
                    files=os.path.join(topk_t,filename_t)
                    #print(files)
                    files_open=open(files,'r')
                    x=1
                    for line in files_open:
                        
                        #print(line)
                        disease_prediction=[]
                        lines=line.strip().split('\t')
                        for d_p in lines:
                            if len(d_p)>1:
                                
                                d_ps=d_p.split(',')
                                #d_ps[0]=id_disease[str(d_ps[0])]
                                d_ps[0]=str(d_ps[0])
                                d_ps[1]=float(d_ps[1])
                                disease_prediction.append(tuple(d_ps))
                        drug=str(x)
                        result_katz[filename][times[0]][drug]=disease_prediction

                        x=x+1

                    files_open.close()
                if method[0]=='rwrh':
                    result_rwrh[filename][times[0]]={}
                    
                    files=os.path.join(topk_t,filename_t)
                    files_open=open(files,'r')
                    x=1
                    for line in files_open:
                        disease_prediction=[]
                        lines=line.strip().split('\t')
                        for d_p in lines:
                            if len(d_p)>1:
                                d_ps=d_p.split(',')
                                #d_ps[0]=id_disease[str(d_ps[0])]
                                d_ps[0]=str(d_ps[0])
                                d_ps[1]=float(d_ps[1])
                                disease_prediction.append(tuple(d_ps))
                        drug=str(x)
                        result_rwrh[filename][times[0]][drug]=disease_prediction
                        x=x+1
                    files_open.close()
                    
                    
              
    for clf in clf_panduan:
        testing_katz_score[clf]={}
        testing_katz_label[clf]={}
        testing_rwrh_score[clf]={}
        testing_rwrh_label[clf]={}
        for t in range(1,T+1):
            times=str(t)
            testing_katz_score[clf][times]=[]
            testing_katz_label[clf][times]=[]
            testing_rwrh_score[clf][times]=[]
            testing_rwrh_label[clf][times]=[]
            testing_t=testing_list[filename][times[0]]
            for test_drug_dis in testing_t:
                drug_t=test_drug_dis[0]
                disease_t=test_drug_dis[1]
                label_t=int(test_drug_dis[2])
                if drug_t in result_katz[clf][times].keys():
                    dis_pre=result_katz[clf][times][drug_t]
                    for d_p in dis_pre:
                        if d_p[0]==disease_t:
                            testing_katz_score[clf][times].append(float(d_p[1]))
                            testing_katz_label[clf][times].append(label_t)
                
                if drug_t in result_rwrh[clf][times].keys():
                    dis_pre=result_rwrh[clf][times][drug_t]
                    for d_p in dis_pre:
                        if d_p[0]==disease_t:
                            testing_rwrh_score[clf][times].append(float(d_p[1]))
                            testing_rwrh_label[clf][times].append(label_t)
    return testing_katz_score,testing_katz_label,testing_rwrh_score,testing_rwrh_label

def transform_alabo2_roman_num(one_num):
    '''
    将阿拉伯数字转化为罗马数字
    '''
    num_list=[1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    str_list=["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    res=''
    for i in range(len(num_list)):
        while one_num>=num_list[i]:
            one_num-=num_list[i]
            res+=str_list[i]
    return res

def get_results(clf,test_svm_prediction_proba,test_svm_pre_label,testing_label,testing_katz_score,testing_katz_label,testing_rwrh_score,testing_rwrh_label):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    tprs_plus = []
    aucs_plus = []
    mean_fpr_plus = np.linspace(0,1,100)
    tprs_katz = []
    aucs_katz = []
    mean_fpr_katz = np.linspace(0,1,100)
    tprs_rwrh = []
    aucs_rwrh = []
    mean_fpr_rwrh = np.linspace(0,1,100)
    for i in range(1,T+1):
        times=str(i)
    
        fpr,tpr,threshold = roc_curve(testing_label[times], test_svm_prediction_proba[times]) ###计算真正率和假正率，fpr,tpr,thresholds 分别为假正率、真正率和阈值。
        
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)##计算auc的值
        aucs.append(roc_auc)
        

        fpr_katz,tpr_katz,threshold = roc_curve(testing_katz_label[times], testing_katz_score[times]) ###计算真正率和假正率，fpr,tpr,thresholds 分别为假正率、真正率和阈值。
        
        tprs_katz.append(interp(mean_fpr_katz, fpr_katz, tpr_katz))
        roc_auc_katz = auc(fpr_katz, tpr_katz)##计算auc的值
        aucs_katz.append(roc_auc_katz)
        
        fpr_rwrh,tpr_rwrh,threshold = roc_curve(testing_rwrh_label[times], testing_rwrh_score[times]) ###计算真正率和假正率，fpr,tpr,thresholds 分别为假正率、真正率和阈值。
        
        tprs_rwrh.append(interp(mean_fpr_rwrh, fpr_rwrh, tpr_rwrh))
        roc_auc_rwrh = auc(fpr_rwrh, tpr_rwrh)##计算auc的值
        aucs_rwrh.append(roc_auc_rwrh)
        
            
    if clf=='1':
        clf_type=transform_alabo2_roman_num(1)
        
        ax=plt.subplot(131)
    
   
        plt.plot([0,1],[0,1],linestyle = '--',lw = 5.0,color = 'navy')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        mean_tpr_katz = np.mean(tprs_katz, axis=0)
        mean_auc_katz = auc(mean_fpr_katz, mean_tpr_katz)
        mean_tpr_rwrh = np.mean(tprs_rwrh, axis=0)
        mean_auc_rwrh = auc(mean_fpr_rwrh, mean_tpr_rwrh)
        
        
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean Metapath2vec ROC (AUC = %0.2f )' % (mean_auc),lw=5.0, alpha=1)
        
        
        plt.plot(mean_fpr_katz, mean_tpr_katz, color='yellow',
                 label=r'Mean katz ROC (AUC = %0.2f )' % (mean_auc_katz),lw=5.0, alpha=1)
        
        plt.plot(mean_fpr_rwrh, mean_tpr_rwrh, color='green',
                 label=r'Mean RWRH ROC (AUC = %0.2f )' % (mean_auc_rwrh),lw=5.0, alpha=1)
        
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.legend(fontsize=20)
        plt.title('%s'%(clf_type),fontsize=30)

        
        plt.legend(loc="lower right")
    if clf=='2':
        clf_type=transform_alabo2_roman_num(2)
        
        ax = plt.subplot(132)
    
   
        plt.plot([0,1],[0,1],linestyle = '--',color = 'navy',linewidth=5.0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        mean_tpr_katz = np.mean(tprs_katz, axis=0)
        mean_auc_katz = auc(mean_fpr_katz, mean_tpr_katz)
        mean_tpr_rwrh = np.mean(tprs_rwrh, axis=0)
        mean_auc_rwrh = auc(mean_fpr_rwrh, mean_tpr_rwrh)
        
        
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(mean_fpr, mean_tpr, color='blue',linewidth=5.0,
                 label=r'Mean Metapath2vec ROC (AUC = %0.2f )' % (mean_auc), alpha=1)
        
        
        plt.plot(mean_fpr_katz, mean_tpr_katz, color='yellow',linewidth=5.0,
                 label=r'Mean katz ROC (AUC = %0.2f )' % (mean_auc_katz),alpha=1)
        
        plt.plot(mean_fpr_rwrh, mean_tpr_rwrh, color='green',linewidth=5.0,
                 label=r'Mean RWRH ROC (AUC = %0.2f )' % (mean_auc_rwrh), alpha=1)
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.legend(fontsize=20)
        plt.title('%s'%(clf_type),fontsize=30)

        plt.legend(loc="lower right")
    if clf=='3':
        clf_type=transform_alabo2_roman_num(3)
        
        ax =plt.subplot(133)
    
   
        plt.plot([0,1],[0,1],linestyle = '--',lw = 5,color = 'navy')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        mean_tpr_katz = np.mean(tprs_katz, axis=0)
        mean_auc_katz = auc(mean_fpr_katz, mean_tpr_katz)
        mean_tpr_rwrh = np.mean(tprs_rwrh, axis=0)
        mean_auc_rwrh = auc(mean_fpr_rwrh, mean_tpr_rwrh)
        
        
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean Metapath2vec ROC (AUC = %0.2f )' % (mean_auc),lw=5.0, alpha=1)
        
       
        plt.plot(mean_fpr_katz, mean_tpr_katz, color='yellow',
                 label=r'Mean katz ROC (AUC = %0.2f )' % (mean_auc_katz),lw=5.0, alpha=1)
        
        plt.plot(mean_fpr_rwrh, mean_tpr_rwrh, color='green',
                 label=r'Mean RWRH ROC (AUC = %0.2f )' % (mean_auc_rwrh),lw=5.0, alpha=1)
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        ax.set_xlabel('False Positive Rate', fontsize=20)
        ax.set_ylabel('True Positive Rate', fontsize=20)
        ax.legend(fontsize=20)
        plt.title('%s'%(clf_type),fontsize=30)
        
        plt.legend(loc="lower right")

    
         
            

def test_svm_allresults(train_vector,train_label,testing_vector,testing_label,clf_time_bestparameters):
    
    t=len(testing_vector)#多少次实验
    testing_F1_score=[]
    test_svm_prediction_proba={}
    test_svm_pre_label={}
    
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
        test_svm_prediction_proba[times]=test_score
        
        testing_f_score=f1_score(test_label, y_hat)
        testing_F1_score.append(testing_f_score)
       
            
            
    print(testing_F1_score)
    return test_svm_prediction_proba,test_svm_pre_label
        
    
def main():
    data_path='../data/CTD'
    
    read_data(data_path)

    Meta_result_path='../data/CTD/embeddings/right'
    embedding_results=os.listdir(Meta_result_path)
    print(embedding_results)
    metapath2vec='0'
    metapath2vec_plus='1'

    for i in range(1,4):
        clf=str(i)
        training_vector[clf]={}
        training_label[clf]={}
        testing_vector[clf]={}
        testing_label[clf]={}
        clf_time_bestparameters[clf]={}
        training_vector_plus[clf]={}
        training_label_plus[clf]={}
        testing_vector_plus[clf]={}
        testing_label_plus[clf]={}
        testing_list[clf]={}
        clf_time_bestparameters_plus[clf]={}
        for filename in embedding_results: #所有的embeddings的结果
           
            print(filename)
            meta_pre=filename.split('_')
            if len(meta_pre)>1: #判断是否是文件夹
                clf_type=meta_pre[2]

                temp=meta_pre[3].split('.')
                if len(temp)>1: #判断是否是embeddings结果
                    
                    times=temp[0]
                
                    if clf_type==clf:  #判断数据分割类型

                        embedding_dir=os.path.join(Meta_result_path,filename)
                        training_vector[clf_type][times],training_label[clf_type][times],testing_vector[clf_type][times],testing_label[clf_type][times],testing_list[clf][times]=get_training_vector_label(embedding_dir,clf_type,times)
                        clf_time_bestparameters[clf_type][times]=training_svm_parameter(training_vector[clf_type][times],training_label[clf_type][times])

    testing_katz_score,testing_katz_label,testing_rwrh_score,testing_rwrh_label=load_katz_rwrh_result()
    
    for clf in ['1','2','3']:


        test_svm_prediction_proba,test_svm_pre_label=test_svm_allresults(training_vector[clf],training_label[clf],testing_vector[clf],testing_label[clf],clf_time_bestparameters[clf])
        test_pre_score[clf]=test_svm_prediction_proba
        test_pre_label[clf]=test_svm_pre_label
        
    plt.figure(13,figsize=[18,6])
    plt.suptitle('PREDICT Data', x=0.5,y=1.05,fontsize=35)

    
    for clf in ['1','2','3']:    
        get_results(clf,test_pre_score[clf],test_pre_label[clf],testing_label[clf],testing_katz_score[clf],testing_katz_label[clf],testing_rwrh_score[clf],testing_rwrh_label[clf])
        
    plt.tight_layout()
             
                            

if __name__ == "__main__":
      
    
    main()

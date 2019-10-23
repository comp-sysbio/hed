# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:25:36 2019

@author: yang_
"""

import numpy as np
from sklearn.model_selection import KFold
import sys
import os
import random
from collections import Counter

id_drug = dict()
id_disease=dict()

disease_drug = dict()
drug_disease = dict()
disease_drug_alldata=dict()
drug_disease_alldata=dict()
edge_list=[]
drug_num=0
disease_num=0
training=dict()
testing=dict()
training_getmetapath=dict()
training_label=dict()
testing_label=dict()

def read_data():
    fileName = r'../data/MSB/drug_disease.txt'
    
    
    drdifile=open(fileName,'r',encoding='utf-8')

    
    for line in drdifile:
        toks = line.strip().split("\t")
        edge=[]
        if len(toks) == 2:
            a= toks[0]
            p= toks[1]
            edge.append(a)
            edge.append(p)
            edge_list.append(edge)
            
            if p not in disease_drug_alldata:
                disease_drug_alldata[p] = []
            disease_drug_alldata[p].append(a)
            disease_drug_alldata[p]=list(set(disease_drug_alldata[p]))
            if a not in drug_disease_alldata:
                drug_disease_alldata[a] = []
            drug_disease_alldata[a].append(p)
            drug_disease_alldata[a]=list(set(drug_disease_alldata[a]))



def splitDataSet(clf, t,training_getmetapath, training,training_label,testing,testing_label):
    
    disease_num=len(disease_drug_alldata)
    drug_num =len(drug_disease_alldata)
    training_getmetapath[t]=[]
    training_label[t]=[]
    training[t]=[]
    testing[t]=[]
    testing_label[t]=[]
    if clf==1:
        
        outfilename = "../data/MSB/metapath/right"+'%s'%clf
        if not os.path.exists(outfilename): #if not outdir,makrdir
            os.makedirs(outfilename)
        outdirtemp=outfilename+'/'+'%d'%t
        if not os.path.exists(outdirtemp): #if not outdir,makrdir
            os.makedirs(outdirtemp)
        file1=open(outdirtemp+'/'+'training.txt','w')
        file2=open(outdirtemp+'/'+'testing.txt','w')
        for drug in drug_disease_alldata.keys():
            drug_dis=drug_disease_alldata[drug]
            #print(drug_dis)
            d_dis_panduan=[]
            for dis in drug_dis:
                #print(dis)
                p=random.random()
                if p<=0.9:
                    dr_di=drug+'\t'+dis
                    training_getmetapath[t].append(dr_di)
                    training[t].append(dr_di)
                    training_label[t].append(1)
                    #random_dis=str(random.randint(1,disease_num))
                    fanwei_int=list(range(1,disease_num+1))
                    fanwei=[str(i) for i in fanwei_int]
                    random_seed=[i for i in fanwei if i not in d_dis_panduan]
                    random_seeds=[i for i in random_seed if i not in drug_dis]
                    #print(disease_num)
                    random_dis=str(random.sample(random_seeds,1)[0])
                    
                    d_dis_panduan.append(random_dis)
                    training[t].append(drug+'\t'+random_dis)
                    training_label[t].append(0)
                    file1.write(dr_di+'\t'+str(1)+'\n')
                    file1.write(drug+'\t'+random_dis+'\t'+str(0)+'\n')
                else:
                    dr_di=drug+'\t'+dis
                    testing[t].append(dr_di)
                    testing_label[t].append(1)
                    fanwei_int=list(range(1,disease_num+1))
                    fanwei=[str(i) for i in fanwei_int]
                    random_seed=[i for i in fanwei if i not in d_dis_panduan]
                    random_seeds=[i for i in random_seed if i not in drug_dis]
                    random_dis=str(random.sample(random_seeds,1)[0])
                    
                    
                    d_dis_panduan.append(random_dis)
                    testing[t].append(drug+'\t'+random_dis)
                    testing_label[t].append(0)
                    file2.write(dr_di+'\t'+str(1)+'\n')
                    file2.write(drug+'\t'+random_dis+'\t'+str(0)+'\n')
        file1.close()
        file2.close()
    if clf==2:

        outfilename = "../data/MSB/metapath/right"+'%s'%clf
        if not os.path.exists(outfilename): #if not outdir,makrdir
            os.makedirs(outfilename)
        outdirtemp=outfilename+'/'+'%d'%t
        if not os.path.exists(outdirtemp): #if not outdir,makrdir
            os.makedirs(outdirtemp)
        file1=open(outdirtemp+'/'+'training.txt','w')
        file2=open(outdirtemp+'/'+'testing.txt','w')
        for disease in disease_drug_alldata.keys():
            dis_drug=disease_drug_alldata[disease]
            #print(drug_dis)
            d_dis_panduan=[]
            for dr in dis_drug:
                #print(dis)
                p=random.random()
                if p<=0.9:
                    dr_di=dr+'\t'+disease
                    training_getmetapath[t].append(dr_di)
                    training[t].append(dr_di)
                    training_label[t].append(1)
                    #random_dis=str(random.randint(1,disease_num))
                    fanwei_int=list(range(1,drug_num+1))
                    fanwei=[str(i) for i in fanwei_int]
                    random_seed=[i for i in fanwei if i not in d_dis_panduan]
                    random_seeds=[i for i in random_seed if i not in dis_drug]
                    random_dis=str(random.sample(random_seeds,1)[0])
                    
                    d_dis_panduan.append(random_dis)
                    training[t].append(random_dis+'\t'+disease)
                    training_label[t].append(0)
                    file1.write(dr_di+'\t'+str(1)+'\n')
                    file1.write(random_dis+'\t'+disease+'\t'+str(0)+'\n')
                else:
                    dr_di=dr+'\t'+disease
                    testing[t].append(dr_di)
                    testing_label[t].append(1)
                    fanwei_int=list(range(1,drug_num+1))
                    fanwei=[str(i) for i in fanwei_int]
                    random_seed=[i for i in fanwei if i not in d_dis_panduan]
                    random_seeds=[i for i in random_seed if i not in dis_drug]
                    random_dis=str(random.sample(random_seeds,1)[0])
                    
                    
                    d_dis_panduan.append(random_dis)
                    testing[t].append(random_dis+'\t'+disease)
                    testing_label[t].append(0)
                    file2.write(dr_di+'\t'+str(1)+'\n')
                    file2.write(random_dis+'\t'+disease+'\t'+str(0)+'\n')
        file1.close()
        file2.close()
    if clf==3:
        outfilename = "../data/MSB/metapath/right"+'%s'%clf
        if not os.path.exists(outfilename): #if not outdir,makrdir
            os.makedirs(outfilename)
        outdirtemp=outfilename+'/'+'%d'%t
        if not os.path.exists(outdirtemp): #if not outdir,makrdir
            os.makedirs(outdirtemp)
        file1=open(outdirtemp+'/'+'training.txt','w')
        file2=open(outdirtemp+'/'+'testing.txt','w')
        drug_panduan={}
        for edge in edge_list:
            p=random.random()
            dr=edge[0]
            dis=edge[1]
            dr_dis=drug_disease_alldata[dr]
            if dr not in drug_panduan.keys():
                drug_panduan[dr]=[]
            #drug_panduan[dr].append(dis)
            
            if p<=0.9:
                dr_di=dr+'\t'+dis
                training_getmetapath[t].append(dr_di)
                training[t].append(dr_di)
                training_label[t].append(1)
                fanwei_int=list(range(1,disease_num+1))
                fanwei=[str(i) for i in fanwei_int]
                random_seed=[i for i in fanwei if i not in drug_panduan[dr]]
                random_seeds=[i for i in random_seed if i not in dr_dis]
                random_dis=str(random.sample(random_seeds,1)[0])
                    
                drug_panduan[dr].append(random_dis)
                training[t].append(dr+'\t'+random_dis)
                training_label[t].append(0)
                file1.write(dr_di+'\t'+str(1)+'\n')
                file1.write(dr+'\t'+random_dis+'\t'+str(0)+'\n')
            else:
                dr_di=dr+'\t'+dis
                testing[t].append(dr_di)
                testing_label[t].append(1)
                fanwei_int=list(range(1,disease_num+1))
                fanwei=[str(i) for i in fanwei_int]
                random_seed=[i for i in fanwei if i not in drug_panduan[dr]]
                random_seeds=[i for i in random_seed if i not in dr_dis]
                random_dis=str(random.sample(random_seeds,1)[0])
                    
                drug_panduan[dr].append(random_dis)
                testing[t].append(dr+'\t'+random_dis)
                testing_label[t].append(0)
                file2.write(dr_di+'\t'+str(1)+'\n')
                file2.write(dr+'\t'+random_dis+'\t'+str(0)+'\n')
        file1.close()
        file2.close()
    return training
                        
   

def get_metapath(training,clf,t):
    numwalks = 1000
    walklength = 100
    
#    outfilename = "..\data\Fdata\metapath"
    drdictfile=open("../data/MSB/id_drug_bianhao.txt",'r',encoding='gbk')
    #with open(dirpath + "/id_author.txt") as adictfile:
        #print(adictfile)
    for line in drdictfile:
        #print(line)
        toks = line.strip().split("\t")
        #print(toks)
        if len(toks) == 2:
            id_drug[toks[0]] = toks[1]
        else:
            print (toks)
            
            
    didictfile=open("../data/MSB/id_disease_bianhao.txt",'r',encoding='gbk')
    #with open(dirpath + "/id_author.txt") as adictfile:
        #print(adictfile)
    for line in didictfile:
        #print(line)
        toks = line.strip().split("\t")
        #print(toks)
        if len(toks) == 2:
            id_disease[toks[0]] = toks[1]
        else:
            print (toks)
   
    for line in training:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            a, p= toks[0], toks[1]
            if p not in disease_drug:
                disease_drug[p] = []
            disease_drug[p].append(a)
            if a not in drug_disease:
                drug_disease[a] = []
            drug_disease[a].append(p)
    
    
    
    
    generate_random_aca(numwalks, walklength,clf,t)
    

 
def generate_random_aca(numwalks, walklength,clf,t):

    outfilename = "../data/MSB/metapath/right"+'%s'%clf
    if not os.path.exists(outfilename): #if not outdir,makrdir
        os.makedirs(outfilename)
    ''' generate the APA metapath '''
    outdirtemp=outfilename+'/'+'MSB_metapath_'+'%s'%t+'.txt'
   
    outfile = open(outdirtemp, 'w',encoding='gbk')
    
    for drug in drug_disease:
        diseases=drug_disease[drug]
        for j in range(0, numwalks ): #wnum walks
            outline = id_drug[drug]
            for i in range(0, walklength):
                nump=len(diseases)
                diseaseid=random.randrange(nump)
                disease=diseases[diseaseid]
                outline+=" "+id_disease[disease]
                drugs=disease_drug[disease]
                numa=len(drugs)
                drugid=random.randrange(numa)
                drug_af=drugs[drugid]
                outline+=" "+id_drug[drug_af]
            outfile.write(outline + "\n")
    outfile.close()


def main():
    

    
    TT=10 #实验的次数
    read_data()
    #tt=0
    for clf in range(1,4): #三种划分数据集形式的判断
        training_getmetapath[clf]={}
        training_label[clf]={}
        training[clf]={}
        testing[clf]={}
        testing_label[clf]={}
        for t in range(1,TT+1):
          
    
            training_get_metapath=splitDataSet(clf,t,training_getmetapath[clf], training[clf],training_label[clf],testing[clf],testing_label[clf])

            get_metapath(training_get_metapath[t],clf,t)
            

if __name__ == '__main__':
    main()
    
    
    
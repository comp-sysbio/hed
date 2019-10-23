clear,clc;
data_path='../data/MSB/metapath/right';
clf_list=dir(data_path);
drug_drug_sim=load('../data/MSB/drug_drug_similarity.txt');
disease_disease_sim=load('../data/MSB/disease_disease_similarity.txt');
%% �Ա��㷨��Ԥ��ǰ200

disease_num = length(disease_disease_sim);

l=disease_num;
t=1:10;%ʵ��Ĵ���
top_200_katz=cell(3,1);
top_200_RWRH=cell(3,1);
testing=cell(3,1);
x=1;

for i=3:length(clf_list)
    top_200_katz{x,1}={};
    top_200_RWRH{x,1}={};
    testing{x,1}={};
%     clf_t=str2num(clf_list(i).name);
%     [a,b]=find(t(1,:)==clf_t);
%     if length(b)>0
        
    t_list=fullfile(data_path,clf_list(i).name);
    t_file=dir(t_list);
    y=1;
    for j =3:length(t_file)
        test=[];
        if t_file(j).isdir==1
            clf_t=str2num(t_file(j).name);
            [a,b]=find(t(1,:)==clf_t);
            if length(b)>0
                
                sublistpath=fullfile(t_list,t_file(j).name);
                training_testing_path=dir(sublistpath);
                t_file(j).name

                 for w =3:length(training_testing_path)

                    if isequal(training_testing_path(w).name,'testing.txt')
                        testingpath=fullfile(sublistpath,training_testing_path(w).name);
                        testing{x,1}{y,1}=load(testingpath); 
                        test=testing{x,1}{y,1};
                    end

                end

                [ScoreMatrix,top_k]=katz(testing{x,1}{y,1},drug_drug_sim,disease_disease_sim,l);
                [top_k_RWRH]=RWRH(testing{x,1}{y,1},drug_drug_sim,disease_disease_sim,l);
                top_200_katz{x,1}{y,1}=top_k;
                top_200_RWRH{x,1}{y,1}=top_k_RWRH;
                y=y+1;
            end
        end
        
    end
    x=x+1;
end

 
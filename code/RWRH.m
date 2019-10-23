function [top_k]=RWRH(training,drug_drug_sim,disease_disease_sim,k)
    MG=drug_drug_sim;
    MP=disease_disease_sim;
    drug_num=length(MG);
    disease_num=length(MP);
    drug_disease_mat=zeros(drug_num,disease_num);
    for i = 1:length(training)
        drug_disease_mat(training(i,1),training(i,2))=1;
    end
    
    MGP=drug_disease_mat;
    [drug_size,disease_size]=size(drug_disease_mat);

    seed=unique(training(:,1));

    %seed=[1:length(drug_size)];
    lamda=0.7 %��ת����
    eta=0;   %��ʼ���ӽڵ������������ռ��
    gama=0.7 %��������
    max_num_iterations=10000; 


    %%����ת�Ƹ���
    MG_transfer=zeros(length(MG));
    for i=1:length(MG)
        sum_i=sum(MG(i,:),2);
        for j=1:length(MG)
            if sum_i~=0
                MG_transfer(i,j)=(1-lamda)*MG(i,j)/sum_i;
            end
        end
    end

    MP_transfer=zeros(length(MP));
    for i=1:length(MP)
        sum_i=sum(MP(i,:),2);
        for j=1:length(MP)
            if sum_i~=0
                MP_transfer(i,j)=(1-lamda)*MP(i,j)/sum_i;
            end
        end
    end

    MGP_transfer=zeros(length(MG),length(MP));
    for i=1:length(MG)
        sum_i=sum(MGP(i,:),2);
        for j=1:length(MP)
            if sum_i~=0
                MGP_transfer(i,j)=lamda*MGP(i,j)/sum_i;
            end
        end
    end

    A=[MG_transfer,MGP_transfer];
    B=[MGP_transfer',MP_transfer];
    M=[A;B];
    seed_RWR=cell(drug_size,1);

    for i=1:length(seed)
        %i
        p0=zeros(drug_size+disease_size,1);
        p0(seed(i,1),1)=1;
        F=p0;
        for j=1:max_num_iterations
            F_pre=F;
            F=(1-gama)*M'*F_pre+gama*p0;
            if ((max(max(abs(F - F_pre)))) <1e-5)
                break
            end
        end
        seed_RWR{seed(i,1),1}=F;    
    end

    %%��RWRH�������
    top_k=cell(drug_size,1);
    for i=1:length(seed_RWR)
        if length(seed_RWR{i,1})>0
            dd=seed_RWR{i,1}(drug_size+1:disease_size+drug_size,1);
            [a,dis]=sort(dd,'descend');
            top_k{i,1}(:,1)=dis(1:k);
            top_k{i,1}(:,2)=a(1:k);
            
        end
    end
end



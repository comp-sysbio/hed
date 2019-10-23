%% Compute Katz scores for (drug, disease) pairs

function [ScoreMatrix,top_k]=katz(training,drug_drug_sim,disease_disease_sim,k)
    l_G = 1;
    l_P = 1;
    beta = 10^(-3);
    iterations = 4;
    c=-15;
    d=9.210340371976184;
    
	DiseaseSimilaritiesLog = 1 ./ (1 + exp(c * disease_disease_sim + d));
    
    %½«training ×ª»»Îª Ï¡Êè¾ØÕó
    drug_num=length(drug_drug_sim);
    disease_num=length(disease_disease_sim);
    A=zeros(drug_num,disease_num);
    for i =1:length(training)
        A(training(i,1),training(i,2))=1;
    end
    training=sparse(A);
    
    for i=1:size(training,2)
        if(sum(training(:,i)) > 0)
            training(:,i) = training(:,i)/norm(training(:,i));
        end
    end
    PP = training*training'; 
    
    display('Computing the powers');
    powersToBeAdded = {};
    GH = drug_drug_sim * training;
    PPH = PP * training;
    GPPH = drug_drug_sim * PPH;
    PPGH =  PP * drug_drug_sim * training; 
    HQH = training * DiseaseSimilaritiesLog * training';
    HQ2 = training * DiseaseSimilaritiesLog * DiseaseSimilaritiesLog;
    GHQ = GH * DiseaseSimilaritiesLog;
    powersToBeAdded{1} = l_G * drug_drug_sim * training;
    powersToBeAdded{2} = l_P^2 * PPH + l_G^2 * drug_drug_sim * GH + GHQ + HQ2;
    powersToBeAdded{3} = l_G^3 * drug_drug_sim * (drug_drug_sim * (drug_drug_sim * training)) + l_G*l_P^2* (GPPH + PPGH) + GHQ * DiseaseSimilaritiesLog + HQ2*DiseaseSimilaritiesLog;
    powersToBeAdded{4} = l_G^4 * drug_drug_sim * (drug_drug_sim * (drug_drug_sim * (drug_drug_sim * training))) + ... 
    l_G^2*l_P^2*(drug_drug_sim * GPPH + drug_drug_sim * PPGH + PP * drug_drug_sim * drug_drug_sim * training) + l_P^4*(PP * PPH) + ... 
    HQH * drug_drug_sim * training + drug_drug_sim * HQH * training + HQ2 * training' * training;
    display('Computing the Katz scores');
    ScoreMatrix = full(beta * powersToBeAdded{1});
    for i = 2:iterations
        ScoreMatrix = ScoreMatrix + beta^i*powersToBeAdded{i};
    end
    display('Done.');

    top_k=cell(drug_num,1);
    for i=1:size(ScoreMatrix,1)
        tem=ScoreMatrix(i,:);
        tem=tem';
        [v,ix]=sort(tem,'descend');
        for j=1:k
            top_k{i,1}(j,1)=ix(j,1);
            top_k{i,1}(j,2)=v(j,1);
        end
    end
 end


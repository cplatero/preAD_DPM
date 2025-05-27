function multi_RPDPM_preAD_v1
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    multi_RPDPM_preAD_v1.m
% Description: Script for training the RPDPM model 
%              on longitudinal neuropsychological data.
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================
clc;
close all;
addpath('./rpdpm');
addpath('./aux_dpm');

%% Input data file

%  Neuropsychological measures:
    % 01 - RAVLT_forgetting (+)
    % 02 - RAVLT_immediate (-) 
    % 03 - RAVLT_learning (-)
    % 04 - RAVLT_perc_forgetting(+)
    % 05 - ADAS13 (+)
    % 06 - FAQ (+)
    % 07 - MMSE (-)
    % 08 - CDRSB (+)
    % 09 - LDELTOTAL (-) 
    % 10 - PACCtrailsB (-) 
    % 11 - TRABSCOR (+)

head_fich_name ="./data_multi/preAD_";
%btstrp=10;
btstrp=2;
classes = {'CN', 'MCI'};
vector_range = 2:5;
porcentaje_nan=0;


load('./data/data_preAD_model_multi_new','y','ages','labels','diagn','subjects',...
    'idx_train','idx_test','feature_names','ranges');



mask_nan_visits=isnan(ages);


%% Pool marker sets    
features_set=1:size(y,1);
max_dim=vector_range(end);
features_subset_matrix = nan(10000,max_dim);
next_row=1;
for i = 1 : length(vector_range)
    aux_subsets = nchoosek(features_set,vector_range(i));
    features_subset_aux=nan(size(aux_subsets,1),max_dim);
    features_subset_aux(:,1:vector_range(i))=aux_subsets;
    n_rows=size(features_subset_aux,1);
    features_subset_matrix(next_row:n_rows+next_row-1,:)=features_subset_aux;
    next_row=next_row+n_rows;
end
features_subset_matrix(next_row:end,:)=[];

%% Read models
alreadyBuiltDPM = false;
if alreadyBuiltDPM
    mask_DPM_built=false(size(features_subset_matrix,1),1);
    for i=1:size(features_subset_matrix,1)
        idx_markers=features_subset_matrix(i,:);
        idx_markers=idx_markers(~isnan(idx_markers));
        str_feat=strrep(int2str(idx_markers), ' ', '_');
        fich_name=head_fich_name+str_feat;
        if(exist(fich_name+'.mat','file'))
            mask_DPM_built(i)=true;
        end
    end
    features_subset_matrix=features_subset_matrix(mask_DPM_built,:);
else
end

%% Performance models
mean_MAE=nan(size(features_subset_matrix,1),length(features_set),2);
std_MAE= nan(size(features_subset_matrix,1),length(features_set),2);
mean_NMAE=nan(size(features_subset_matrix,1),2);
std_NMAE= nan(size(features_subset_matrix,1),2); 
mean_AUC=nan(size(features_subset_matrix,1),2);
std_AUC= nan(size(features_subset_matrix,1),2);

mean_percent_sCU=nan(size(features_subset_matrix,1),2);
std_percent_sCU=nan(size(features_subset_matrix,1),2);
mean_percent_sMCI=nan(size(features_subset_matrix,1),2);
std_percent_sMCI=nan(size(features_subset_matrix,1),2);
mean_percent_pCU=nan(size(features_subset_matrix,1),2);
std_percent_pCU=nan(size(features_subset_matrix,1),2);

mean_corr_MCI_age=nan(size(features_subset_matrix,1),2);
std_corr_MCI_age=nan(size(features_subset_matrix,1),2);
mean_corr_MCI_reserve=nan(size(features_subset_matrix,1),2);
std_corr_MCI_reserve=nan(size(features_subset_matrix,1),2);

str_feat=string.empty(size(features_subset_matrix,1), 0);

%% Building DPMs
% pc = parcluster('local');
% pc.JobStorageLocation = strcat(getenv('SCRATCH'),'/', getenv('SLURM_JOB_ID'));
% parpool(pc, str2num(getenv('SLURM_CPUS_ON_NODE')));


for i = 1 : size(features_subset_matrix,1)
    idx_markers=features_subset_matrix(i,:);
    idx_markers=idx_markers(~isnan(idx_markers));
    model_feat=strjoin(feature_names(idx_markers), ', ');
    fprintf('Model %d of %s,\n', i, model_feat);
    select_feat = add_nan_features(y(idx_markers,:,:),~mask_nan_visits,porcentaje_nan);
    str_feat(i)=strrep(int2str(idx_markers), ' ', '_');
    fich_name=head_fich_name+str_feat(i)+ '_'+num2str(porcentaje_nan)+'_nan';
    if(~exist(fich_name+'.mat','file'))
        [MAE,NMAE,AUC,percent_sCU,percent_sMCI,percent_pCU,corr_MCI_age,corr_MCI_reserve]=...
            build_rpdpm_preAD(select_feat,ages,labels,diagn,...
            classes,ranges(idx_markers,:),idx_train,idx_test,btstrp,...
            fich_name,feature_names(idx_markers));
    else
        load(fich_name+'.mat', 'auc','MAE','NMAE','m_test','m_train');
        AUC=auc;
        percent_sCU=[m_test.percentage_sCU,m_train.percentage_sCU];
        percent_sMCI=[m_test.percentage_sMCI,m_train.percentage_sMCI];
        percent_pCU=[m_test.percentage_pCU,m_train.percentage_pCU];
        corr_MCI_age=[m_test.corr_MCI_age,m_train.corr_MCI_age];
        corr_MCI_reserve=[m_test.corr_MCI_reserve,m_train.corr_MCI_reserve];
        
    end

        
    mean_NMAE(i,:)=mean(NMAE);
    std_NMAE(i,:)=std(NMAE);
    mean_MAE(i,idx_markers,:)=mean(MAE);
    std_MAE(i,idx_markers,:)=std(MAE);
    mean_AUC(i,:)=mean(AUC);
    std_AUC(i,:)=std(AUC);

    mean_percent_sCU(i,:)=mean(percent_sCU);
    std_percent_sCU(i,:)=std(percent_sCU);
    mean_percent_sMCI(i,:)=mean(percent_sMCI);
    std_percent_sMCI(i,:)=std(percent_sMCI);
    mean_percent_pCU(i,:)=mean(percent_pCU);
    std_percent_pCU(i,:)=std(percent_pCU);


    mean_corr_MCI_age(i,:)=mean(corr_MCI_age);
    std_corr_MCI_age(i,:)=std(corr_MCI_age);
    mean_corr_MCI_reserve(i,:)=mean(corr_MCI_reserve);
    std_corr_MCI_reserve(i,:)=std(corr_MCI_reserve);
        
    fprintf('\n');
end
    
    

%% Overall results
results=table;
results.str_feat=str_feat';
results.mean_percent_sCU=mean_percent_sCU;
results.mean_percent_sMCI=mean_percent_sMCI;
results.mean_percent_pCU=mean_percent_pCU;
results.mean_corr_MCI_age=mean_corr_MCI_age;
results.mean_corr_MCI_reserve=mean_corr_MCI_reserve;
results.mean_AUC=mean_AUC;

save('Metrics_multi_NaN_0','mean_MAE','std_MAE','mean_NMAE','std_NMAE','std_AUC',...
    'std_percent_sCU','std_percent_sMCI','std_percent_pCU',...
    'std_corr_MCI_age','std_corr_MCI_reserve','results',...
    'feature_names','vector_range');




%% End
% delete(gcp('nocreate'));

end




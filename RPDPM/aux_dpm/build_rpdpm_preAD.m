function [MAE,NMAE,auc,percentage_sCU,percentage_sMCI,percentage_pCU,corr_MCI_age,corr_MCI_reserve]=...
    build_rpdpm_preAD(y,ages,labels,diagnose,classes,ranges,...
    idx_train,idx_test,btstrp,fich_name,feat_names)
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    build_rpdpm_preAD.m
% Description: Building DPMs using RPDPM
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================
%% Optimization parameters
loss_type = 'logistic'; % robust estimation loss function type
fit_type = 'proposed'; % fitting function type
L_max = 50; % maximum number of alternating iterations
L_min = 10; % minimum number of alternating iterations

%% data
y_train = y(:, idx_train, :);
ages_train = ages(idx_train, :);
labels_train = labels(idx_train, :);
diagnose_train = diagnose(:, idx_train);

y_test = y(:, idx_test, :);
ages_test = ages(idx_test, :);
labels_test = labels(idx_test, :);


%% Robust Parametric DPM Training (model)
% Optimization options
optim_options = optimoptions('lsqnonlin', 'Display', 'none', 'Jacobian', 'off', 'DerivativeCheck', 'off', 'MaxIter', 1e+3, 'TolFun', 1e-6, 'TolX', 1e-6, 'Algorithm', 'trust-region-reflective');


[Sigma, A, B, C, D, G, DPS_train, Idx_train, idx_valid]=...
    build_model_RPDPM(y_train,ages_train,labels_train,diagnose_train,...
    classes,ranges,loss_type, fit_type, optim_options,btstrp,L_max,L_min);


%% Robust Parametric DPM Testing
fprintf('Test \t');
[DPS_test,NMAE_test,MAE_test]=get_MAE_RPDPM_v1(y_test,feat_names,ages_test,Sigma,A,B,C,D,G,loss_type, fit_type, optim_options,btstrp);


%% Robust Parametric DPM train
fprintf('Train \t');
[DPS_train_2,NMAE_train,MAE_train]=get_MAE_RPDPM_v1(y_train,feat_names,ages_train,Sigma,A,B,C,D,G,loss_type, fit_type, optim_options,btstrp);

%% Robust Parametric DPM Classification Training
% Training Bayesian classifiers per bootstrap
UC = numel(classes); % number of unique class labels
dist_models = repmat({'kernel'}, 1, UC); % distribution models for likelihoods
posterior_bayes = cell(UC, btstrp);
likelihood_bayes = cell(UC, btstrp);
evidence_bayes = cell(1, btstrp);
prior_bayes = zeros(UC, btstrp);
for n = 1 : btstrp
    [posterior_bayes(:, n), likelihood_bayes(:, n), prior_bayes(:, n), evidence_bayes{n}] = bayes_train(DPS_train(:, :, n), labels_train(Idx_train(:, n), :), classes, dist_models);
end

%% Robust Parametric DPM Classification Testing
fprintf('Test \t');
auc_test=get_AUC_RPDPM_v1(DPS_test,posterior_bayes,labels_test,classes,btstrp);
[ages_onset,gt_onset,m_test]=get_Onset_MCI_preAD(DPS_test,...
    posterior_bayes,ages_test,labels_test,btstrp);
onset_test=[ages_onset,gt_onset];


%% Robust Parametric DPM Classification Validation
fprintf('Train \t');
auc_train=get_AUC_RPDPM_v1(DPS_train_2,posterior_bayes,labels_train,classes,btstrp);
[ages_onset,gt_onset,m_train]=get_Onset_MCI_preAD(DPS_train_2,...
    posterior_bayes,ages_train,labels_train,btstrp);
onset_train=[ages_onset,gt_onset];

%% Metrics
auc=[auc_test',auc_train'];
NMAE=[NMAE_test,NMAE_train];
MAE(:,:,1)=MAE_test';
MAE(:,:,2)=MAE_train';

percentage_sCU=[m_test.percentage_sCU,m_train.percentage_sCU];
percentage_sMCI=[m_test.percentage_sMCI,m_train.percentage_sMCI];
percentage_pCU=[m_test.percentage_pCU,m_train.percentage_pCU];

corr_MCI_age=[m_test.corr_MCI_age,m_train.corr_MCI_age];
corr_MCI_reserve=[m_test.corr_MCI_reserve,m_train.corr_MCI_reserve];


%% save model
save(fich_name,'Sigma','A','B','C','D','G','loss_type', ...
    'fit_type', 'optim_options','btstrp','auc','MAE','NMAE',...
    'onset_test','onset_train','m_test','m_train',...
    'DPS_train', 'Idx_train', 'idx_valid');

end





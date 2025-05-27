function auc_test=get_AUC_RPDPM_v1(DPS_test,posterior_bayes,labels_test,classes,btstrp)
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    get_AUC_RPDPM_v1.m
% Description: Script for evaluating the RPDPM model (AUC)
%              on longitudinal neuropsychological data.
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================
%% Robust Parametric DPM Classification Testing
% Classification performance analysis for test samples
UC = numel(classes); % number of unique class label
[I,J,~]=size(DPS_test);
n_visits=I*J;
predict_scores_test = nan(UC, n_visits, btstrp); % posterior probabilities for test scores
auc_test = zeros(1, btstrp); % test multiclass area under the curve per bootstrap
for n = 1 : btstrp
    dps_test_n = DPS_test(:, :, n);
    for cc = 1 : UC
        predict_scores_test(cc, :, n) = posterior_bayes{cc, n}(dps_test_n(:)');
    end
    auc_test(n) = multiclass_auc(predict_scores_test(:, :, n)', labels_test(:), classes);
end
auc_test_avg = mean(auc_test); % average multiclass area under the curve
auc_test_std = std(auc_test); % standard deviation of multiclass areas under the curves
fprintf(' AUC = %4.4f \x00B1 %4.4f \n', [auc_test_avg, auc_test_std]); % display test classification performance
end

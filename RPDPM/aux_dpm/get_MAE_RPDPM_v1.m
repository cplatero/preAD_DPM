function [DPS_test,NMAE_test,MAE_test]=...
    get_MAE_RPDPM_v1(y_test,feat_names,ages_test,Sigma,A,B,C,D,G,loss_type, fit_type, optim_options,btstrp) 
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    get_AUC_RPDPM_v1.m
% Description: Script for evaluating the RPDPM model (MAE)
%              on longitudinal neuropsychological data.
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================
%% Robust Parametric DPM Testing

% Estimating test subject parameters using biomaker values and ages
[K2, I2, J2] = size(y_test); % number of biomarkers, subjects, and visits
y_test_fit = zeros(K2, I2, J2, btstrp); % estimated values of biomarkers
DPS_test = zeros(I2, J2, btstrp);
Alpha_test = zeros(I2, btstrp);
Beta_test = zeros(I2, btstrp);
for n = 1 : btstrp
    [Alpha_test(:, n), Beta_test(:, n), DPS_test(:, :, n), y_test_fit(:, :, :, n)] = rpdpm_test(y_test, ages_test, ...
        Alpha_test(:, n), Beta_test(:, n), Sigma(:, n), A(:, n), B(:, n), C(:, n), D(:, n), G(:, n), loss_type, fit_type, optim_options);

end

% Test set modeling performance
NMAE_test = zeros(btstrp, 1); % overall test error per bootstrap
MAE_test = zeros(K2, btstrp); % test error per biomarker per bootstrap
for n = 1 : btstrp
    ae_test = abs(y_test - y_test_fit(:, :, :, n)); % absolute errors
    MAE_test(1 : K2, n) = nanmean(ae_test(1 : K2, :), 2); % mean absolute error
    ae_test = ae_test(1 : K2, :) ./ Sigma(1 : K2, n); % normalized (scaled) absolute errors
    NMAE_test(n) = nanmean(ae_test(:)); % normalized mean absolute error
end

for k=1:K2
        fprintf('MAE (%s): %.3f\t',feat_names(k),mean(MAE_test(k,:,end),'omitnan'));
end

nmae_test_avg = mean(NMAE_test); % average of normalized mean absolute errors
nmae_test_std = std(NMAE_test); % standard deviation of normalized mean absolute errors
fprintf('\nNMAE = %4.4f \x00B1 %4.4f \n', [nmae_test_avg, nmae_test_std]); % display test modeling performance

end

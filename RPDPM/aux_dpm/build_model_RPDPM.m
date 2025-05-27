function [Sigma, A, B, C, D, G, DPS_train, Idx_train,idx_valid]=...
    build_model_RPDPM(y_train,ages_train,labels_train,diagnose_train,classes,ranges,loss_type, fit_type, optim_options,btstrp,L_max,L_min)
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    get_AUC_RPDPM_v1.m
% Description: Script for training the RPDPM model
%              on longitudinal neuropsychological data.
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================
% Number of biomarkers, subjects, and visits
[K1, I1, J1] = size(y_train);


% Parameter initialization for different bootstraps
Idx_train = zeros(I1, btstrp);
Alpha_train = zeros(I1, btstrp);
Beta_train = zeros(I1, btstrp);
Sigma = zeros(K1, btstrp);
A = zeros(K1, btstrp);
B = zeros(K1, btstrp);
C = zeros(K1, btstrp);
D = zeros(K1, btstrp);
G = zeros(K1, btstrp);
BIC = zeros(L_max, btstrp);
Iters_opt = zeros(1, btstrp);
DPS_train = zeros(I1, J1, btstrp);
Loss_train = zeros(L_max, btstrp);
Loss_valid = zeros(L_max, btstrp);
NMAE_valid = zeros(L_max, btstrp);

% Biomarker modeling
num_points = sum(sum(~isnan(y_train), 3), 1); % number of available points per training subject
for n = 1 : btstrp
    
    fprintf('bootstrap %i> ', n); % display bootstrap number
    
    % Randomly sampling a group of within-class subjects with replacement for training and the rest for validation
    idx_btstrp = []; % indices of bootstrapped subjects
    rng(n); % random number generation seed
    for u = 1 : numel(classes)
        for uu = u : numel(classes)
            idx_u = find(strcmp(diagnose_train(1, :), classes(u)) & strcmp(diagnose_train(2, :), classes(uu))); % first and last diagnoses
            med_u = median(num_points(idx_u)); % within class median of available points
            idx_many = find(num_points(idx_u) > med_u); % subjects with many points
            if numel(idx_many)
                idx_rnd = randi(numel(idx_many), numel(idx_many), 1);
                idx_btstrp = cat(2, idx_btstrp, idx_u(idx_many(idx_rnd)));
            end
            idx_few = find(num_points(idx_u) <= med_u); % subjects with few points
            if numel(idx_few)
                idx_rnd = randi(numel(idx_few), numel(idx_few), 1);
                idx_btstrp = cat(2, idx_btstrp, idx_u(idx_few(idx_rnd)));
            end
        end
    end
    Idx_train(:, n) = idx_btstrp; % indices of training subjects
    idx_valid = find(~ismember(1 : I1, idx_btstrp)); % indices of validation subjects
    
    % Fitting sigmoids to biomarkers
    [idx_train, idx_btstrp1, idx_btstrp2] = unique(idx_btstrp); % unique training indices
    [alpha, beta, Sigma(:, n), a, b, c, d, g, Loss_train(:, n), BIC(:, n)] = ...
        rpdpm_train(y_train(:, idx_train, :), ages_train(idx_train, :), labels_train(idx_train, :), classes, ranges, loss_type, fit_type, L_max, optim_options);
    
    % Validation set modeling performance
    w = zeros(numel(idx_valid), J1, K1);
    for i = 1 : numel(idx_valid)
        w(i, :) = 1 / sum(sum(~isnan(y_train(:, idx_valid(i), :)), 3), 1); % normalized w.r.t. the number of available points per subject
    end
    alpha_valid = zeros(numel(idx_valid), 1);
    beta_valid = zeros(numel(idx_valid), 1);
    for l = 1 : L_max
        theta = [a(:, l + 1), b(:, l + 1), c(:, l + 1), d(:, l + 1), g(:, l + 1)];
        [alpha_valid, beta_valid, ~, y_valid_fit] = rpdpm_test(y_train(:, idx_valid, :), ages_train(idx_valid, :), ...
            alpha_valid, beta_valid, Sigma(:, n), theta(:, 1), theta(:, 2), theta(:, 3), theta(:, 4), theta(:, 5), loss_type, fit_type, optim_options);
        Loss_valid(l, n) = sum(objective_markers(theta(:), y_train(:, idx_valid, :), ages_train(idx_valid, :), w(:), Sigma(:, n), ...
            repmat(alpha_valid, 1, J1), repmat(beta_valid, 1, J1), loss_type, fit_type) .^ 2);
        ae_valid = abs(y_train(:, idx_valid, :) - y_valid_fit); % absolute errors
        ae_valid(1 : K1, :) = ae_valid(1 : K1, :) ./ Sigma(1 : K1, n); % normalized (scaled) absolute errors
        NMAE_valid(l, n) = nanmean(ae_valid(:)); % normalized mean absolute errors
    end
    
    % Optimal parameters
    [~, L_opt] = min(flip(Loss_valid(L_min : end, n))); % the latest minimum
    Iters_opt(n) = L_max - L_opt + 1;
    alpha = alpha(:, Iters_opt(n) + 1);
    beta = beta(:, Iters_opt(n) + 1);
    A(:, n) = a(:, Iters_opt(n) + 1);
    B(:, n) = b(:, Iters_opt(n) + 1);
    C(:, n) = c(:, Iters_opt(n) + 1);
    D(:, n) = d(:, Iters_opt(n) + 1);
    G(:, n) = g(:, Iters_opt(n) + 1);
    dps_ = repmat(beta, 1, J1) + repmat(alpha, 1, J1) .* ages_train(idx_train, :); % linear DPS
    
    % Standardizing DPS and related parameters w.r.t. the normal group
    dps_cn = dps_(cellfun(@(x) strcmp(x, classes{1}), labels_train(idx_train, :))); % normal visits
    mu_cn = nanmean(dps_cn(:)); % mean
    sigma_cn = nanstd(dps_cn(:)); % standard deviation
    alpha = alpha / sigma_cn; % normalized alpha
    beta = (beta - mu_cn) / sigma_cn; % normalized beta
    dps = (dps_ - mu_cn) / sigma_cn; % normalized DPS
    C(:, n) = (C(:, n) - mu_cn) / sigma_cn; % normalized c
    B(:, n) = B(:, n) * sigma_cn; % normalized b
    
    % Replicated parameter values for all bootstrapped subjects
    Alpha_train(:, n) = alpha(idx_btstrp2);
    Beta_train(:, n) = beta(idx_btstrp2);
    DPS_train(:, :, n) = dps(idx_btstrp2, :);
    
end
bic_avg = mean(BIC(logical(full(sparse(Iters_opt, 1 : btstrp, ones(1, btstrp)))))); % average BIC
bic_std = std(BIC(logical(full(sparse(Iters_opt, 1 : btstrp, ones(1, btstrp)))))); % standard deviation of BICs
nmae_valid_avg = mean(NMAE_valid(logical(full(sparse(Iters_opt, 1 : btstrp, ones(1, btstrp)))))); % average of normalized mean absolute errors
nmae_valid_std = std(NMAE_valid(logical(full(sparse(Iters_opt, 1 : btstrp, ones(1, btstrp)))))); % standard deviation of normalized mean absolute errors
fprintf('Training BIC = %4.4f \x00B1 %4.4f \n', [bic_avg, bic_std]); % display training modeling performance
fprintf('Validation NMAE = %4.4f \x00B1 %4.4f \n', [nmae_valid_avg, nmae_valid_std]); % display validation modeling performance
end

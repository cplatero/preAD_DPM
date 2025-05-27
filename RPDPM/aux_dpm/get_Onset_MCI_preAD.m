function [ages_onset,gt_onset,metrics]=get_Onset_MCI_preAD(DPS_test,posterior_bayes,ages_test,labels_test,btstrp)
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    get_AUC_RPDPM_v1.m
% Description: Script for evaluating the RPDPM model (clinical scores)
%              on longitudinal neuropsychological data.
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================

I=size(DPS_test,1);
ages_onset=nan(I,btstrp);
for n = 1 : btstrp
    dps_test_n = DPS_test(:, :, n);
    prob_MCI=posterior_bayes{2, n}(dps_test_n)>.5;
    for i=1: I
        if(sum(prob_MCI(i,:)>0))
            mask_values= ~isnan(ages_test(i,:));
            age_subj=ages_test(i,mask_values);
            label_subj=prob_MCI(i,mask_values);
            idx=find(label_subj);
            if(label_subj(end))
                if(idx(1)>1)
                    ages_onset(i,n)=(age_subj(idx(1))+age_subj(idx(1)-1))/2;
                else
                    ages_onset(i,n)=age_subj(idx(1));
                end
            end
        end %convert
    end %subjects
end %btstrp

gt_onset=get_GT_onset(labels_test,ages_test);

%% metrics
% sCU + sMCI
age_bsl=ages_test(:,1);

metrics=table;
percentage_sCU=nan(btstrp,1);
percentage_sMCI=nan(btstrp,1);
percentage_pCU=nan(btstrp,1);
corr_MCI_age=nan(btstrp,1);
corr_MCI_reserve=nan(btstrp,1);

num_sCU=sum(isnan(gt_onset));
num_sMCI=sum(gt_onset==age_bsl);

for n = 1 : btstrp
    percentage_sCU(n)=sum(isnan(ages_onset(:,n)) & isnan(gt_onset))/num_sCU*100;
    percentage_sMCI(n)=sum((ages_onset(:,n)==age_bsl) & (gt_onset==age_bsl))/num_sMCI*100;
end

% pCU
num_pCU=sum(gt_onset>age_bsl);

for n = 1 : btstrp
    percentage_pCU(n)=sum((ages_onset(:,n)>age_bsl) & (gt_onset>age_bsl))/num_pCU*100;
    cov=corrcoef(rmmissing([ages_onset(:,n),gt_onset]));
    corr_MCI_age(n)=cov(1,2);
    cov=corrcoef(rmmissing([ages_onset(:,n)-age_bsl,gt_onset-age_bsl]));
    corr_MCI_reserve(n)=cov(1,2);
    
end

metrics.percentage_sCU=percentage_sCU;
metrics.percentage_sMCI=percentage_sMCI;
metrics.percentage_pCU=percentage_pCU;
metrics.corr_MCI_age=corr_MCI_age;
metrics.corr_MCI_reserve=corr_MCI_reserve;

%% show
fprintf('Percentage sCU: %.1f (%.2f)\n',mean(metrics.percentage_sCU),...
    std(metrics.percentage_sCU));
fprintf('Percentage sMCI: %.1f (%.2f)\n',mean(metrics.percentage_sMCI),...
    std(metrics.percentage_sMCI));
fprintf('Percentage pCU: %.1f (%.2f)\n',mean(metrics.percentage_pCU),...
    std(metrics.percentage_pCU));

fprintf('Correlation age conversion: %.2f (%.2f)\n',mean(metrics.corr_MCI_age),...
    std(metrics.corr_MCI_age));
fprintf('Correlation cognition reserve: %.2f (%.2f)\n',mean(metrics.corr_MCI_reserve),...
    std(metrics.corr_MCI_reserve));


end









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gt_onset=get_GT_onset(labels_test,ages_test)
I=size(labels_test,1);
gt_onset=nan(I,1);
MCI=strcmp(labels_test,'MCI');

for i=1: I
    if(sum(MCI(i,:)>0))
        mask_values= ~isnan(ages_test(i,:));
        age_subj=ages_test(i,mask_values);
        label_subj=MCI(i,mask_values);
        idx=find(label_subj);
        if(label_subj(end))
            if(idx(1)>1)
                gt_onset(i)=(age_subj(idx(1))+age_subj(idx(1)-1))/2;
            else
                gt_onset(i)=age_subj(idx(1));
            end
        end
    end %convert
end %subjects

end
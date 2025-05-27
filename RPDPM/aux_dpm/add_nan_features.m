function matriz=add_nan_features(matriz,mask_visits,porcentaje_nan)
% ============================================================
% Project:    Disease progression modeling from early AD stage
% Repository: https://github.com/cplatero/preAD_DPM
% Author:     Carlos Platero
% Email:      carlos.platero@upm.es
% Institution:Universidad Politécnica de Madrid 
% ------------------------------------------------------------
% Filename:    add_nan_features.m
% Description: Robustness to missing data
% 
% Version:    1.0
% Date:       2025-05-09
% MATLAB Ver: R2024a 
% ============================================================

% Count the total elements and the existing NaNs
n_features=size(matriz,1);
num_features = sum(mask_visits(:))*n_features;
num_nan_actuales=0;
for i=1:n_features
    feature_=squeeze(matriz(i,:,:));
    feature_=feature_(mask_visits);
    num_nan_actuales=num_nan_actuales+sum(isnan(feature_));
end


% Calculate how many values ​​should be converted to NaN
num_nan_deseados = round(num_features * porcentaje_nan / 100);
num_nan_a_agregar = round((num_nan_deseados - num_nan_actuales)/n_features);

fprintf('Missing data %d (%.2f), adding nan %d (%.2f)\n',...
    num_nan_actuales,num_nan_actuales/num_features*100,...
    num_nan_deseados,num_nan_deseados/num_features*100);

% If there are already enough NaNs, do nothing.
if num_nan_a_agregar > 0

    % Get the indexes of non-NaN values
    for i=1:n_features
        feature_=squeeze(matriz(i,:,:));
        feature_1D=feature_(mask_visits);
        indices_no_nan = find(~isnan(feature_(mask_visits)));

        if(i<n_features)
            feature_next=squeeze(matriz(i+1,:,:));
            feature_1D_next=feature_next(mask_visits);
            indices_nan_next = find(isnan(feature_1D_next));
            indices_no_nan =setdiff(indices_no_nan,indices_nan_next);
        end
        if(i>1)
            feature_prev=squeeze(matriz(i-1,:,:));
            feature_1D_prev=feature_prev(mask_visits);
            indices_nan_prev = find(isnan(feature_1D_prev));
            indices_no_nan =setdiff(indices_no_nan,indices_nan_prev);
        end
        

        % Select indexes randomly to convert to NaN
        indices_aleatorios = randsample(indices_no_nan, num_nan_a_agregar);
        feature_1D(indices_aleatorios) = NaN;
        feature_(mask_visits)=feature_1D;
        matriz(i,:,:)=feature_;
    end
   
end

end
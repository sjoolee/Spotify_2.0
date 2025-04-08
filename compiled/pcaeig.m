function [T, P, R2, R2_All] = pcaeig(X, C) %X = data, C = chosen # of components

    % normalizing the dataset
    means = mean(X);
    dev = std(X - means);
    norm_data = (X - means)./dev; % centered and scaled dataset

    % finds the eigenvectors (V) and eigenvalues (D) from X_T * X operation
    XTX = norm_data'*norm_data;
    [V, D] = eig(XTX); 

    % orders in standard notation, descending eigenvalue with corresponding
    % eigenvector
    V = fliplr(V); % 1st col is p1, 2nd col is p2...
    D = rot90(D,2);

    P = V(:, 1:C); % picks the chosen number of components
    T = norm_data*P; % The transpose of P is not required (as seen in lecture)
    % since eig(XTX) produces the eigenvectors in column form, not row form
    
    d = diag(D); % grabs the eigenvalues which correspond to variance described
    R2=0;
    % for-loop uses the eigenvalues to find the summed explained variance
    % notably: all eigenvalues are used for the sum division, but we're
    % only summing R2 up to the chosen number of components to see (out of
    % 1) how much variance is explained using X-component PCA
    for i = 1:C
        R2 = R2 + d(i)/sum(d);
    end

    R2_All = zeros(1,(width(X)));
    for i = 1:width(X)
        R2_All(i) = d(i)/sum(d);
    end 
end
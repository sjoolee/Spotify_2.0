function [idx, centroids] = KMeans(data, k)
    % Inputs:
    %   data:     (n x m) matrix where n = number of data points, m = number of features
    %   k:        number of clusters
    %   maxIter:  maximum number of iterations

    maxIter = 100;

    [n, m] = size(data);

    % Step 1: Randomly initialize centroids
    rng('default'); % For reproducibility
    randIdx = randperm(n, k);
    centroids = data(randIdx, :);

    % Initialize variables
    idx = zeros(n, 1);         % Cluster assignments
    prevCentroids = centroids; % For checking convergence

    for iter = 1:maxIter
        % Step 2: Assign each point to the nearest centroid
        for i = 1:n
            distances = sum((centroids - data(i,:)).^2, 2);
            [~, idx(i)] = min(distances);
        end

        % Step 3: Update centroids
        for j = 1:k
            clusterPoints = data(idx == j, :);
            if ~isempty(clusterPoints)
                centroids(j, :) = mean(clusterPoints, 1);
            end
        end

        % Step 4: Check for convergence
        if isequal(prevCentroids, centroids)
            disp(['Converged in ', num2str(iter), ' iterations.']);
            break;
        end
        prevCentroids = centroids;
    end
end
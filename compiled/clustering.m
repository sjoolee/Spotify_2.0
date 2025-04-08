%%official project working 
%% Acquire Data and Plot Matrix
clc
clear
close all

%read files
preprocess_music = readmatrix("data/cleaned_X.csv");
C = 6; % change to select number of components
[T, P, ~, ~] = pcaeig(preprocess_music, C);
labels = readcell('data/post_drop_features.csv');
labels = string(labels);
% F1 = loading_plot(P(:,1),1,labels); %acousticness, energy, loudness, valence - peppiness 
% F1 = loading_plot(P(:,2),3,labels); % danceability, duration, instrumentalness - chill vibes - musicness?
% F1 = loading_plot(P(:,3),3,labels); %speechiness, danceability, liveness - yappage
% F1 = loading_plot(P(:,4),4,labels);

%plottign against 
T1 = T(:, 1);
T2 = T(:, 2);
T3 = T(:, 3);

% figure;
% sz = 15;
% scatter(T1, T2, sz, "filled");
% xlabel('Scores of PC1');
% ylabel('Scores of PC2');
% title("Score Plot of PC1 and PC2");
% 
% figure;
% sz = 15;
% scatter(T1, T3, sz, "filled");
% xlabel('Scores of PC1');
% ylabel('Scores of PC3');
% title("Score Plot of PC1 and PC3");
% 
% figure;
% sz = 15;
% scatter(T2, T3, sz, "filled");
% xlabel('Scores of PC2');
% ylabel('Scores of PC3');
% title("Score Plot of PC2 and PC3");


%clustering
% Assume T is your PCA score matrix (e.g., T(:,1:3))
T_reduced = T(:, 1:3);

% Cluster using custom k-means
k = 6;  % Number of clusters
[idx, centroids] = KMeans(T_reduced, k);

% Plot results (3D example)
figure
scatter3(T_reduced(:,1), T_reduced(:,2), T_reduced(:,3), 50, idx, 'filled');
hold on;
scatter3(centroids(:,1), centroids(:,2), centroids(:,3), 100, 'kx', 'LineWidth', 2);
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('K-Means Clustering (From Scratch)');
grid on;

%%generating a playlist

% New point to add (also 3D)
length(preprocess_music)
random_point = 20
test_point = preprocess_music(random_point,:)*P

% Number of points to retrieve
num_closest_points = 20;

% Calculate Euclidean distances between the new point and all existing points
%distances = sqrt((abs(preprocess_music)).^2 + (abs(test_point)).^2);
distances = sqrt(sum((T_reduced - test_point(1,1:3)).^2,2));

% Sort the distances in ascending order and get the indices of the closest points
[sorted_distances, sorted_indices] = sort(distances);

% Retrieve the closest 20 points (or less if there are fewer than 20 points)
closest_indices = sorted_indices(1:num_closest_points)

% Get the closest points
closest_points = preprocess_music(closest_indices, :);

%plot
figure
scatter3(T_reduced(:,1), T_reduced(:,2), T_reduced(:,3), 5, idx, 'filled');
hold on;
scatter3(centroids(:,1), centroids(:,2), centroids(:,3), 100, 'kx', 'LineWidth', 2);
hold on

% plot the new point (random_point could be an index OR a separate point)
if isscalar(random_point) % it's an index into T_reduced
    scatter3(T_reduced(random_point,1), T_reduced(random_point,2), T_reduced(random_point,3), 3000, 'rx', "LineWidth", 10);
else % it's a new 1x3 vector (not yet in T_reduced)
    scatter3(random_point(1), random_point(2), random_point(3), 100, 'rx', 'LineWidth', 1);
end

% plot the closest points
for k = 1:length(closest_indices)
    idx_k = closest_indices(k);
    scatter3(T_reduced(idx_k,1), T_reduced(idx_k,2), T_reduced(idx_k,3), 100, 'k*');
end

legend('Data points', 'Centroids', 'New Point', 'Closest Points')

%fetch name of recommended songs
Music = readtable("data/cleaned_mega.csv");
Titles = Music (2:end,2:4);
Test_song = Titles(random_point,:)
Recommended_songs = Titles(closest_indices,:)
%%
iter = 30;  % Number of cluster values to test
inertia = zeros(iter,1);

for k = 1:iter
    [idx, centroids] = KMeans(T_reduced, k);
    total_dist = 0;

    for j = 1:length(idx)
        cluster_id = idx(j);
        d = norm(T_reduced(j,:) - centroids(cluster_id,:))^2;  % squared distance
        total_dist = total_dist + d;
    end

    inertia(k) = total_dist;
end

% Plot elbow
figure
plot(1:iter, inertia, '-o', 'LineWidth', 2);
xlabel('Number of Clusters (k)');
ylabel('Inertia (WCSS)');
title('Elbow Plot for Optimal k');
grid on;
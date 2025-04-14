%% Grabs the specified PCA
clear; clc; close all;
filename = "cleaned_mega.csv"; music = readmatrix(filename);
% X = [music(2:end, 10:13) music(2:end, 15) music(2:end, 17:22)];
X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)]; % mega
% X = [music(2:end, 4) music(2:end, 7:8) music(2:end, 10) music(2:end, 12:14) music(2:end, 16:17)]; % yearly

rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);

X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech

means = mean(X);
dev = std(X - means);

X = preprocess(X);
C = 3;
[T, P, R2, ~] = pcaeig(X, C);

% writematrix(P, 'data/mega/mega_P.csv');
%% projects each user song to Mega Playlist PC space
filename = "genre_cleaned.csv"; music = readmatrix(filename);
% X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)]; % mega
% X2 = [music(1:end, 4) music(1:end, 7:8) music(1:end, 10) music(1:end, 12:14) music(1:end, 16:17)]; % yearly
X2 = music(:, 5:13);
X2(:,1) = sqrt(X2(:,1)); % duration
X2(:,3) = log1p(X2(:,3)); % energy
X2(:,5) = sqrt(X2(:,5)); % speech

norm_data = (X2 - means)./dev;

T2 = norm_data*P;

writematrix(T2, 'data/2024-2025/projected_T.csv');
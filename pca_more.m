
%% PCAs all possible combinatiosn
clear; clc; close all;
% music, X range, and threshold are the only things adjustable
% music = readmatrix("cleaned2020-2022.csv");
% X = music(:, 5:19);
music = readmatrix("musicals_playlist.csv");
X = music(:, 10:20);
threshold = 0.8;

% automatic
rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);

num_features = width(X);
total_combinations = 2^num_features - 1;

% Generate binary matrix for all subsets (Each row = a feature subset)
% omg this is 2di or 2dx binary combinations but using features as the OMFG
% it starts at the LAST feature, and works it's way upwards to the FIRST
% feature
feature_matrix = dec2bin(1:total_combinations) - '0';
R2_vals = zeros(size(feature_matrix));
num_selected = zeros(length(feature_matrix),1);

for i = 1:total_combinations %total_combinations
    selected_features = logical(feature_matrix(i, :));
    num_selected(i) = sum(selected_features);
    if (num_selected(i) == 1) % skips when only 1 component is used
        continue
    end
    data_subset = X(:, selected_features);

    C = 1; % doesn't matter what is picked
    [~, ~, ~, R2_All] = prob1_fun(data_subset, C);
    for j = 1:length(R2_All)
        R2_vals(i,j) = R2_All(j);
    end
end
%% gets min num of components required to get R2 > 80 per PCA combo
min_comp_tot_comp = zeros(length(feature_matrix),2);
for i = 1:total_combinations
    sum = 0;
    for j = 1:num_features
        if R2_vals(i,j) == 0
            break
        end

        sum = sum + R2_vals(i,j);
        if sum >= threshold
            min_comp_tot_comp(i,1) = j;
            min_comp_tot_comp(i,2) = num_selected(i);
            break
        end
    end
end
%% finds best PCA combos which reduces the most components
differences = abs(min_comp_tot_comp(:,2) - min_comp_tot_comp(:,1));
max(differences)
ind = find(differences == max(differences));

found = zeros(length(ind), 2);
for i = 1:length(ind)
    found(i,:) = min_comp_tot_comp(ind(i),:);
end
%% finds the minimum number of total components
% less total components at this stage means the dimension reduced has a
% higher percentage reduction compared to a PCA combination with more total
% components. Ex. 6 out of 12 is 50% reduced but 7 out of 13 is 46.1538% reduced

% the corresponding min_ind indices need to be checked with the og ind 
% variable which is using the original datasets combinations
min_ind = find(found(:,2) == min(found(:,2)))
Final_combinations = ind(min_ind)
%%
labels = ["Duration" "Popularity" "Danceability" "Energy" "Key" "Loudness" "Mode" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature" "Song Age"];
dropped = strings(length(Final_combinations), length(labels));
for i = 1:length(Final_combinations)
    selected_features = logical(feature_matrix(Final_combinations(i), :));
    select = labels(~selected_features);
    for j=1:length(labels(~selected_features))
        dropped(i,j) = select(j);
    end
end
%% does not work without 
C = Final_combinations(1);
[T, P, R2, R2_All] = prob1_fun(array2table(prod), C);

reconstructed = T*P';
x = reconstructed;
t = norm_data_qual;

net = feedforwardnet(1);
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;

net = train(net,x',t');
y = net(x');

close all;
scatter(t,y)
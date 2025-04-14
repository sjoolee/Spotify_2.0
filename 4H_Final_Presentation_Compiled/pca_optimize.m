%% lines that need to be changed per dataset:
% 7-8 depending on dataset and chosen cols
% 79 to update column names, and 98, 107, 108 to pick the right components
% 246 to change file name
%% PCAs all possible combinatiosn
clear; clc; close all;
filename = "data/mega/genre_cleaned.csv"; music = readmatrix(filename);
% X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)]; % mega
X = [music(2:end, 4) music(2:end, 7:8) music(2:end, 10) music(2:end, 12:14) music(2:end, 16:17)]; % yearly
X = music(:, 5:13);

rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);

X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech
% X = music(2:end, 10:23);
threshold = 0.8;

X = preprocess(X);
%%
num_features = width(X);
total_combinations = 2^num_features - 1;

% Generate binary matrix for all subsets (Each row = a feature subset)
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
    [~, ~, ~, R2_All] = pcaeig(data_subset, C);
    for j = 1:length(R2_All)
        R2_vals(i,j) = R2_All(j);
    end
end
% gets min num of components required to get R2 > 80 per PCA combo
min_comp_tot_comp = zeros(length(feature_matrix),2);
for i = ceil(total_combinations/2):total_combinations
    sum1 = 0;
    for j = 1:num_features
        if R2_vals(i,j) == 0
            break
        end

        sum1 = sum1 + R2_vals(i,j);
        if sum1 >= threshold
            min_comp_tot_comp(i,1) = j;
            min_comp_tot_comp(i,2) = num_selected(i);
            break
        end
    end
end
% finds best PCA combos which reduces the most components
differences = abs(min_comp_tot_comp(1:end-1,2) - min_comp_tot_comp(1:end-1,1));
max(differences)
ind = find(differences == max(differences));

found = zeros(length(ind), 2);
for i = 1:length(ind)
    found(i,:) = min_comp_tot_comp(ind(i),:);
end
% finds the maximum number of total componenets to prevent excessive
% dropping
max_ind = find(found(:,2) == max(found(:,2)));
Final_combinations = ind(max_ind);

% labels change per dataset, need to be careful
% labels = ["Duration" "Popularity" "Danceability" "Energy" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo"];
labels = ["Duration" "Danceability" "Energy" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Valence" "Tempo"];
% labels = ["Duration" "Popularity" "Danceability" "Energy" "Key" "Loudness" "Mode" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature"];
dropped = strings(length(Final_combinations), length(labels));
for i = 1:length(Final_combinations)
    selected_features = logical(feature_matrix(Final_combinations(i), :));
    select = labels(~selected_features);
    for j=1:length(labels(~selected_features))
        dropped(i,j) = select(j);
    end
end

R2s = zeros(length(Final_combinations), 1); % finds max R2
for i = 1:length(Final_combinations)
    selected_features = logical(feature_matrix(Final_combinations(i), :)); % chooses 1 row of the found combos
    data_subset = X(:, selected_features);
    C = 3; % change to select number of components
    [~, ~, R2, ~] = pcaeig(data_subset, C);
    R2s(i) = R2;
end
max(R2s)
i_choose = find(R2s == max(R2s));

P_choose = 0;
T_choose = 0;
for i = 1:length(Final_combinations) % uses max R2 to get best combination
    selected_features = logical(feature_matrix(Final_combinations(i), :)); % chooses 1 row of the found combos
    data_subset = X(:, selected_features);
    C = 3; % change to select number of components
    [T, P, R2, ~] = pcaeig(data_subset, C);
    R2s(i) = R2;
    if (i == i_choose) %the best explained variance iteration,changes per dataset
        P_choose = P;
        T_choose = T;
        fprintf("%s\n", labels(selected_features))
        disp("\n");
        fprintf("%s\n", labels(~selected_features))
        break
    end
end

C = found(i_choose,1);
%selected_features = logical(feature_matrix(Final_combinations(i_choose), :)); % chooses 1 row of the found combos
selected_features = logical(feature_matrix(end, :));
X = X(:, selected_features);
og = readtable(filename, VariableNamingRule="preserve");
%% T2 cleaning 
[t, ~, ~] = nipalspca(X,5);
T2 = zeros(length(t),1);
for i = 1:length(t)
    for j = 1:size(t,2)
        T2(i) = T2(i) + (t(i,j)/std(t(:,j)))^2;
    end
end
A = 9;
thing = finv(0.95, A,length(t)-A);
LIMIT = ((length(t)-1)*(length(t)+1)*A)/length(t)/(length(t)-A)*thing;

thing2 = finv(0.99, A,length(t)-A);
LIMIT2 = ((length(t)-1)*(length(t)+1)*A)/length(t)/(length(t)-A)*thing2;

figure;
plot(T2,'k-');
hold on;
plot([0 length(t)],[LIMIT LIMIT],'r--');
plot([0 length(t)],[LIMIT2 LIMIT2],'g--');
xlabel('Tablet Observation')
ylabel('Hotelling''s T^2');
axis([0 length(t) 0 max(T2)]);
%% SPE cleaning
[t, p, ~] = nipalspca(preprocess(X),5);
X_HAT = t*p';
res = preprocess(X) - X_HAT;

% calculate SPE
N = size(res,1);
SE = res.^2;
SPE = sum(SE,2);

% get degrees of freedom
v = std(SPE)^2;
m = mean(SPE);
df = (2*m^2)/v;

% calculate SPE limits
SPE95lim = (v/(2*m))*chi2inv(0.95,df);
SPE99lim = (v/(2*m))*chi2inv(0.99,df);

% plot SPE along with confidence limits
F = figure;
x = 1:N;
y95 = ones(1,N)*SPE95lim;
y99 = ones(1,N)*SPE99lim;

plot(x,SPE,'ko-')
hold on
plot(x,y95,'--r')
plot(x,y99,'-r')

box on;
grid on;

xlabel('Observation')
ylabel('Squared Prediction Error (SPE)')
legend('SPE Values','95% Limit','99% Limit')

hold off;

for i = 1:width(x) % removes common outliers
    if (SPE(i) > y99(1) && T2(i) > LIMIT2)
        disp(x(i))
        X(i,:) = nan;
    end
end

rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);
SPE = SPE(rowsToKeep, :);
og = og(rowsToKeep, :);

[t, p, ~] = nipalspca(X,C);
X_HAT = t*p';
res = preprocess(X) - X_HAT;
N = size(res,1);
x = 1:N;
y95 = ones(1,N)*SPE95lim;
y99 = ones(1,N)*SPE99lim;

figure
plot(x,SPE,'ko-')
hold on
plot(x,y95,'--r')
plot(x,y99,'-r')

box on;
grid on;

xlabel('Observation')
ylabel('Squared Prediction Error (SPE)')
legend('SPE Values','95% Limit','99% Limit')

hold off;
%%
C = 3; % change to select number of components
% X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)];
[T, P, R2, R2_All] = pcaeig(X, C);
% writematrix(X, 'data/cleaned/cleaned_mega.csv');
writetable(og, 'data/2024-2025/cleaned_data.csv', 'WriteVariableNames', true)
writematrix(T, 'data/2024-2025/cleaned_Tscores.csv');
writematrix(X, 'data/2024-2025/cleaned_X.csv');
% writematrix(labels(selected_features), 'data/post_drop_features.csv');
%% scree plot
clear; clc; close all;
labels = ["Duration" "Danceability" "Energy" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Valence" "Tempo"];
C = 3; % change to select number of components
filename = 'cleaned_mega.csv'; music = readmatrix(filename);
X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)];
X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech
[T, P, R2, R2_All] = pcaeig(X, C);

r2_scree = zeros(1,length(R2_All));
for i = 1:length(r2_scree)
    r2_scree(i) = sum(R2_All(1:i));
end

str = '#1BD75F';
color = sscanf(str(2:end), '%2x%2x%2x', [1 3]) / 255;
plot(1:i, r2_scree, Color=color, Marker="o", LineWidth=1.5)

hold on

filename = "genre_cleaned.csv"; music = readmatrix(filename);
X = music(:, 5:13);
X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech
[T, P, R2, R2_All] = pcaeig(X, C);

r2_scree = zeros(1,length(R2_All));
for i = 1:length(r2_scree)
    r2_scree(i) = sum(R2_All(1:i));
end

str = '#6C6C6D';
color = sscanf(str(2:end), '%2x%2x%2x', [1 3]) / 255;
plot(1:i, r2_scree, Color=color, Marker="+", LineWidth=1.5)
grid("on")
xlabel("Dimensions")
ylabel("Explained Variance")
title("Scree Plot for Mega Playlist and 2024-2025 Listening History")

legend("Mega Playlist", "2024-2025", Location="southeast")
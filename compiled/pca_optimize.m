%% lines that need to be changed per dataset:
% 7-8 depending on dataset and chosen cols
% 79 to update column names, and 98, 107, 108 to pick the right components
% 234 to change file name
%% PCAs all possible combinatiosn
clear; clc; close all;
filename = "data/MEGA_PLAYLIST.csv"; music = readmatrix(filename);
X = music(3:end, 10:22);
threshold = 0.8;

X = preprocess(X);
% rowsWithNaN = any(isnan(X), 2);
% rowsToKeep = ~rowsWithNaN;
% X = X(rowsToKeep, :);
% music = music(rowsToKeep+3, :);

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
    [~, ~, ~, R2_All] = pcaeig(data_subset, C);
    for j = 1:length(R2_All)
        R2_vals(i,j) = R2_All(j);
    end
end
% gets min num of components required to get R2 > 80 per PCA combo
min_comp_tot_comp = zeros(length(feature_matrix),2);
for i = 1:total_combinations
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
differences = abs(min_comp_tot_comp(:,2) - min_comp_tot_comp(:,1));
max(differences)
ind = find(differences == max(differences));

found = zeros(length(ind), 2);
for i = 1:length(ind)
    found(i,:) = min_comp_tot_comp(ind(i),:);
end
% finds the minimum number of total components
% less total components at this stage means the dimension reduced has a
% higher percentage reduction compared to a PCA combination with more total
% components. Ex. 6 out of 12 is 50% reduced but 7 out of 13 is 46.1538% reduced

% the corresponding min_ind indices need to be checked with the og ind 
% variable which is using the original datasets combinations
min_ind = find(found(:,2) == min(found(:,2)));
Final_combinations = ind(min_ind);
%
% labels change per dataset, need to be careful
labels = ["Duration" "Popularity" "Danceability" "Energy" "Key" "Loudness" "Mode" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature"];
dropped = strings(length(Final_combinations), length(labels));
for i = 1:length(Final_combinations)
    selected_features = logical(feature_matrix(Final_combinations(i), :));
    select = labels(~selected_features);
    for j=1:length(labels(~selected_features))
        dropped(i,j) = select(j);
    end
end

R2s = zeros(length(Final_combinations), 1);
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
for i = 1:length(Final_combinations)
    selected_features = logical(feature_matrix(Final_combinations(i), :)); % chooses 1 row of the found combos
    data_subset = X(:, selected_features);
    C = 3; % change to select number of components
    [T, P, R2, ~] = pcaeig(data_subset, C);
    R2s(i) = R2;
    if (i == i_choose) %the best explained variance iteration,changes per dataset
        P_choose = P;
        T_choose = T;
        disp(labels(selected_features))
        disp(labels(~selected_features))
        break
    end
end

C = found(i_choose,1);
selected_features = logical(feature_matrix(Final_combinations(i_choose), :)); % chooses 1 row of the found combos
X = X(:, selected_features);
%% T2 cleaning 
og = readtable(filename);
[t, ~, ~] = nipalspca(X,C);
T2 = zeros(length(t),1);
for i = 1:length(t)
    for j = 1:size(t,2)
        T2(i) = T2(i) + (t(i,j)/std(t(:,j)))^2;
    end
end
A = 4-1;
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

for i = 1:length(X)
    if (T2(i) > LIMIT2)
        % disp(x(i))
        X(i,:) = nan;
        T2(i) = nan;
    end
end

rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);
T2 = T2(rowsToKeep, :);
rowsToKeep = logical([0;1;rowsToKeep]);
og = og(rowsToKeep, :);

figure;
plot(T2,'k-');
hold on;
plot([0 length(t)],[LIMIT LIMIT],'r--');
plot([0 length(t)],[LIMIT2 LIMIT2],'g--');
xlabel('Tablet Observation')
ylabel('Hotelling''s T^2');
axis([0 length(t) 0 max(T2)]);
%% SPE cleaning
[t, p, ~] = nipalspca(X,C);
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

for i = 1:width(x)
    if (SPE(i) > y99)
        disp(x(i))
        X(i,:) = nan;
    end
end

rowsWithNaN = any(isnan(X), 2);
rowsToKeep = ~rowsWithNaN;
X = X(rowsToKeep, :);
SPE = SPE(rowsToKeep, :);
rowsToKeep = logical([1;rowsToKeep]);
og = og(rowsToKeep, :);

[t, p, ~] = nipalspca(X,C);
X_HAT = t*p';
res = preprocess(X) - X_HAT;
N = size(res,1);
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
%%
C = 3; % change to select number of components
[T, P, R2, R2_All] = pcaeig(X, C);
% writematrix(X, 'data/cleaned/cleaned_mega.csv');
writetable(og, 'data/cleaned_mega.csv', 'WriteVariableNames', false)
writematrix(T, 'data/cleaned_Tscores.csv');
writematrix(X, 'data/cleaned_X.csv');
writematrix(labels(selected_features), 'data/post_drop_features.csv');
%% MATLAB driver - info
% Written by: Team 1

% Primarily uses combined_data_dupe which takes play count and replicates
% the row to naturally weight the song

% Some aspects use functions made by Jake Nease "score_loading_plot.m" and
% "loading_plot.m"

% the PCA code used the eigenvalue decomposition function developed by Team
% 1 "prob1_fun.m"
%% PCA
clear; clc; close all;
music = readmatrix("Combined_data_dupe.csv");
prod = music(:, 6:18);
% prod = [music(:, 6:8), music(:, 10:11), music(:, 14:16), music(:, 18)];
rowsWithNaN = any(isnan(prod), 2);
rowsToKeep = ~rowsWithNaN;
qual = music(:,19);
prod = prod(rowsToKeep, :);
qual = qual(rowsToKeep, :);
clc;

means_prod = mean(prod);
dev_prod = std(prod - means_prod);
norm_data_prod = (prod - means_prod)./dev_prod; % centered and scaled dataset

means_qual = mean(qual);
dev_qual = std(qual - means_qual);
norm_data_qual = (qual - means_qual)./dev_qual; % centered and scaled dataset

C = 10;
[T, P, R2, R2_All] = prob1_fun(array2table(prod), C);
R2
R2_All
reconstructed = T*P'; % if C == all columns, then reconstructed should be the same as norm_data_prod

labels = ["Duration" "Popularity" "Danceability" "Key" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature" "Song Age"];
F1 = loading_plot(P(:,1),1,labels);
F2 = loading_plot(P(:,2),2,labels);
F3 = loading_plot(P(:,3),3,labels);

score_loading_plot(T(:,1), T(:,2), P(:,1), P(:,2), labels)
%% ANN with PCA
x = reconstructed;
t = norm_data_qual;

net = feedforwardnet(1);
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;
net = train(net,x',t');
y = net(x');

close all;
scatter(t,y)
%% higher hidden-layers (after post-processing if needed)
close all;
% net2 = fitnet([10, 10, 10]);
net2 = feedforwardnet(10);
net2.divideParam.trainRatio = 0.70;
net2.divideParam.valRatio = 0.15;
net2.divideParam.testRatio = 0.15;
net2.trainParam.epochs = 500;
net2 = train(net2,x',t');
y2 = net2(x');
%% ANN original data: scatterplot + training code
clear; clc; close all;

music = readmatrix ("Combined_data_dupe.csv");
prod = music(:, 6:18);
% prod = [music(:, 6:8), music(:, 10:11), music(:, 14:16), music(:, 18)];
rowsWithNaN = any(isnan(prod), 2);
rowsToKeep = ~rowsWithNaN;
qual = music(:,19);
prod = prod(rowsToKeep, :);
qual = qual(rowsToKeep, :);
clc;

means_prod = mean(prod);
dev_prod = std(prod - means_prod);
norm_data_prod = (prod - means_prod)./dev_prod; % centered and scaled dataset

min_max_prod = (prod - min(prod))/(max(prod) - min(prod));

means_qual = mean(qual);
dev_qual = std(qual - means_qual);
norm_data_qual = (qual - means_qual)./dev_qual; % centered and scaled dataset

min_max_qual = (qual - min(qual))/(max(qual) - min(qual));

x = norm_data_prod;
t = norm_data_qual;

% just to see where the vertical line shows up
net = feedforwardnet(1);
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;
net = train(net,x',t');
y = net(x');

close all;
scatter(t,y)
xlabel("ANN Output")
ylabel("Target (play count)")
title("Scatterplot of Target and Output")
%% after testing data, used to clean vertical line
close all;

t_range = [t(end) - 0.07, t(end) + 0.07];
y_upper = [abs(mean(y)) * 2, 100];
y_lower = [mean(y) * 2, -100];

index_t = true(size(t)); % Create a logical array of true values

for i = 1:length(t)
    if ((t(i) > t_range(1)) && (t(i) < t_range(2)))
        index_t(i) = false;
    end
end

index_y = true(size(y));
for i = 1:length(y)
    if (((y(i) > y_upper(1)) && (y(i) < y_upper(2))) || ((y(i) < y_lower(1)) && (y(i) > y_lower(2))))
        index_y(i) = false;
    end
end

index = index_t | index_y';

t = t(index);
x = x(index);
y = y(index');
scatter(t, y)
%% run this after all above
close all;
net2 = feedforwardnet(10);
net2.divideParam.trainRatio = 0.70;
net2.divideParam.valRatio = 0.15;
net2.divideParam.testRatio = 0.15;
net2.trainParam.epochs = 500;
net2 = train(net2,x',t');
y2 = net2(x');
scatter(t,y2)
xlabel("ANN Output")
ylabel("Target (play count)")
title("Scatterplot of Target and Output (MCS)")
%% trying to manipulate different distributions
close all;
shift = abs(min(norm_data_prod(:,5)));
adjusted = norm_data_prod(:,5) + shift;
transformed = log(adjusted);
histogram(transformed - log(shift))
figure
histogram(norm_data_prod(:,5))
%% plot all histograms separately
histogram(norm_data_prod(:,1)) % music(:, 6);
title("music(:, 6)")
figure
histogram(norm_data_prod(:,2)) % music(:, 7);
title("music(:, 7)")
figure
histogram(norm_data_prod(:,3)) % music(:, 8);
title("music(:, 8)")
figure
histogram(norm_data_prod(:,4)) % music(:, 9); bad
title("music(:, 9)")
figure
histogram(norm_data_prod(:,5)) % music(:, 10); 
title("music(:, 10)")
figure
histogram(norm_data_prod(:,6)) % music(:, 11); 
title("music(:, 11)")
figure
histogram(norm_data_prod(:,7)) % music(:, 12); bad
title("music(:, 12)")
figure
histogram(norm_data_prod(:,8)) % music(:, 13); bad
title("music(:, 13)")
figure
histogram(norm_data_prod(:,9)) % music(:, 14); maybe
title("music(:, 14)")
figure
histogram(norm_data_prod(:,10)) % music(:, 15);
title("music(:, 15)")
figure
histogram(norm_data_prod(:,11)) % music(:, 16); maybe
title("music(:, 16)")
figure
histogram(norm_data_prod(:,12), 50) % music(:, 17); bad
title("music(:, 17)")
figure
histogram(norm_data_prod(:,13)) % music(:, 18);
title("music(:, 18)")
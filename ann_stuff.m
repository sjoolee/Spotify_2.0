%% scatterplot + training code
clear; clc; close all;
% prod = readtable("Combined_Play_Count2020-2022.csv");
% prod = prod(:,7:18);
% qual = readtable("Combined_Play_Count2020-2022.csv");
% qual = qual(:,19);

music_23_24 = readmatrix ("Combined_Play_Count2023-2024.csv");
prod = music_23_24(:,7:18);
rowsWithNaN = any(isnan(prod), 2);
rowsToKeep = ~rowsWithNaN;
qual = music_23_24(:,19);
prod = prod(rowsToKeep, :);
qual = qual(rowsToKeep, :);

clc;

means_prod = mean(prod);
dev_prod = std(prod - means_prod);
norm_data_prod = (prod - means_prod)./dev_prod; % centered and scaled dataset

means_qual = mean(qual);
dev_qual = std(qual - means_qual);
norm_data_qual = (qual - means_qual)./dev_qual; % centered and scaled dataset

x = norm_data_prod;
t = norm_data_qual;

net = feedforwardnet(1);
net.trainParam.epochs = 1000;
net.trainParam.showWindow = false;
net = train(net,x',t');
y = net(x');

close all;
scatter(t,y)

%% run this after all above
close all;
net2 = feedforwardnet(10);
net2.divideParam.trainRatio = 0.70;
net2.divideParam.valRatio = 0.15;
net2.divideParam.testRatio = 0.15;
net2.trainParam.epochs = 500;
net2 = train(net2,x',t');
y2 = net(x');

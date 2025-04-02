%% Project PLS - total combined (no dupes)
% Written by: Team 1

% Uses the combined_final_file dataset that does not contain duplicates but
% a greater number of song tracks (additional processing was required to
% fill in NaN gaps)

% Some aspects use functions made by Jake Nease "loading_plot.m"

% the PCA code used the eigenvalue decomposition function developed by Team
% 1 "prob1_fun.m"
%% Acquire Data and Plot Matrix
clc
clear
close all

%read file
music = readmatrix ("combined_final_file.csv");

%plot matrix - raw
Xraw = music(:,6:19);
plotmatrix(Xraw)
title("Raw")
figure;

%plot matrix - min-max scale
Xmax = max(Xraw);
Xmin = min(Xraw);
X = (Xraw-Xmin)./(Xmax-Xmin);
plotmatrix(X)
title("Min-Max Scale")
figure;

%plot matrix - mcs
Xmean = mean(Xraw);
Xstd = std(Xraw);
X = (Xraw-Xmean)./Xstd;
plotmatrix(X)
title("Mean Center Scale")

%data for analysis
X = music(:,6:18);
Y = music(:,19);
n = 14; %max number of variables to see how R2 changes

%% PLS - MCS
[t_MCS,u_MCS,w_star_MCS,c_MCS,p_MCS,R2_MCS] = PLS_MCS(X,Y,n);

%loading plot
music = readtable ("combined_final_file.csv");
music = music(:,6:18);
data = table2array(music);labels = ["Duration" "Popularity" "Danceability" "Key" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature" "Song Age"];

F1 = loading_plot(w_star_MCS(:,1),1,labels);
F2 = loading_plot(w_star_MCS(:,2),2,labels);
F3 = loading_plot(w_star_MCS(:,3),3,labels);

%score plot 1
T1_MCS = t_MCS(:, 1);
T2_MCS = t_MCS(:, 2);
T3_MCS = t_MCS(:, 3);

figure;
sz = 15;
scatter(T1_MCS, T2_MCS, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC2');
title("Score Plot of PC1 and PC2 - MCS");

%score plt 2
figure;
sz = 15;
scatter(T2_MCS, T3_MCS, sz, "filled");
xlabel('Scores of PC2');
ylabel('Scores of PC3');
title("Score Plot of PC2 and PC3 - MCS");

%score plt 3
figure;
sz = 15;
scatter(T1_MCS, T3_MCS, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC3');
title("Score Plot of PC1 and PC3 - MCS");

%% PLS - Min-Max Scale
[t_MMS,u_MMS,w_star_MMS,c_MMS,p_MMS,R2_MMS]=PLS_Min_Max_Scale(X,Y,n);

%loading plot
music = readtable ("combined_final_file.csv");
music = music(:,6:18);
data = table2array(music);labels = ["Duration" "Popularity" "Danceability" "Key" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature" "Song Age"];

F1 = loading_plot(w_star_MMS(:,1),1,labels);
F2 = loading_plot(w_star_MMS(:,2),2,labels);
F3 = loading_plot(w_star_MMS(:,3),3,labels);

%score plot 1
T1_MMS = t_MMS(:, 1);
T2_MMS = t_MMS(:, 2);
T3_MMS = t_MMS(:, 3);

figure;
sz = 15;
scatter(T1_MMS, T2_MMS, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC2');
title("Score Plot of PC1 and PC2 - MMS");

%score plt 2
figure;
sz = 15;
scatter(T2_MMS, T3_MMS, sz, "filled");
xlabel('Scores of PC2');
ylabel('Scores of PC3');
title("Score Plot of PC2 and PC3 - MMS");

%score plt 3
figure;
sz = 15;
scatter(T1_MMS, T3_MMS, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC3');
title("Score Plot of PC1 and PC3 - MMS");



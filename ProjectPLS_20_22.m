%Project PLS - 20-22
clc
clear
close all

music_20_22 = readmatrix ("Combined_Play_Count2020-2022.csv");
X = music_20_22(:,7:18);
Y = music_20_22(:,19);
n = 3;

[t,u,w_star,c,p,R2] = PLS(X,Y,n)

%plot matrix
Xraw = music_20_22(:,7:19)
Xmean = mean(Xraw);
Xstd = std(Xraw);
X = (Xraw-Xmean)./Xstd;

plotmatrix(X,X)

%loading plot
music_20_22 = readtable ("Combined_Play_Count2020-2022.csv");
music_20_22 = music_20_22(:,7:18)
data = table2array(music_20_22);labels = ["Duration" "Popularity" "Danceability" "Key" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Liveness" "Valence" "Tempo" "Time Signature"];
F1 = loading_plot(w_star(:,1),1,labels)
F2 = loading_plot(w_star(:,2),2,labels)
F3 = loading_plot(w_star(:,3),3,labels)

%score plot 1
T1 = t(:, 1);
T2 = t(:, 2);
T3 = t(:, 3);

figure;
sz = 15;
scatter(T1, T2, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC2');
title("Score Plot of PC1 and PC2");

%score plt 2
figure;
sz = 15;
scatter(T2, T3, sz, "filled");
xlabel('Scores of PC2');
ylabel('Scores of PC3');
title("Score Plot of PC2 and PC3");


%score plt 3
figure;
sz = 15;
scatter(T1, T3, sz, "filled");
xlabel('Scores of PC1');
ylabel('Scores of PC3');
title("Score Plot of PC1 and PC3");
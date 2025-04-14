%%
% Disclaimer: Uses Jake Nease's modified loading_plot code to use subplots
%%
clear; clc; close all;
labels = ["Duration" "Danceability" "Energy" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Valence" "Tempo"];
C = 3; % change to select number of components
filename = 'data/mega/cleaned_mega.csv'; music = readmatrix(filename);
X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)];
X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech
[T, P, R2, R2_All] = pcaeig(X, C);

figure
% subplot(1,3,1)
p_choose = 1;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("Mega Playlist PC1")

figure
% subplot(1,3,2)
p_choose = 2;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("Mega Playlist PC2")

figure
% subplot(1,3,3)
p_choose = 3;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("Mega Playlist PC3")

%% loading vectors for user history
filename = "data/2024-2025/cleaned_data.csv"; music = readmatrix(filename);
X = music(:, 5:13);
X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech
[T, P, R2, R2_All] = pcaeig(X, C);

figure
p_choose = 1;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("2024-2025 Listening History PC1")

figure
p_choose = 2;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("2024-2025 Listening History PC2")

figure
p_choose = 3;
loading_plot(P(:,p_choose), p_choose, labels, gca)
title("2024-2025 Listening History PC3")
%% before and after transformation plotmatrix
clear; clc; close all;
C = 3; % change to select number of components
filename = 'data/mega/cleaned_mega.csv'; music = readmatrix(filename);

figure
labels = ["Dur." "Pop." "Dance" "Energy" "Key" "Loudness" "Mode" "Speech" "Acoustic" "Instr." "Liveness" "Valence" "Tempo" "Time Sig."];
[~,ax] = plotmatrix(music(:,10:23));
iter = size(ax,1);
for i = 1:iter
    ax(i,1).YLabel.String = labels(i);
    ax(iter,i).XLabel.String = labels(i);
end

figure
X = [music(2:end, 10) music(2:end, 12:13) music(2:end, 15) music(2:end, 17:19) music(2:end, 21:22)];
labels = ["Duration" "Danceability" "Energy" "Loudness" "Speechiness" "Acousticness" "Instrumentalness" "Valence" "Tempo"];
[~,ax] = plotmatrix(X);
iter = size(ax,1);
for i = 1:iter
    ax(i,1).YLabel.String = labels(i);
    ax(iter,i).XLabel.String = labels(i);
end

X(:,1) = sqrt(X(:,1)); % duration
X(:,3) = log1p(X(:,3)); % energy
X(:,5) = sqrt(X(:,5)); % speech

figure
[~,ax] = plotmatrix(X);
iter = size(ax,1);
for i = 1:iter
    ax(i,1).YLabel.String = labels(i);
    ax(iter,i).XLabel.String = labels(i);
end
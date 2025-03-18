%% Cleaning Dataset
clear; clc; close all;
dataset = readtable("spotify_data.csv"); % Reads the given .csv file
dataset = removevars(dataset,{'Var1', 'track_id'});

fname = 'Streaming_History_Audio_2022-2023_1.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

structDataTable = struct2table(val);
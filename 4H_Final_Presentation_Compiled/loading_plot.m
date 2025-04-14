% % function that creates a loadings bar plot
% 
% % Jake Nease
% % Chemical Engineering
% % McMaster University
% 
% % Pass in the appropriate loading
% % Pass in the tag of the loading (1, 2, 3 etc)
% % Pass in the names of the data labels

% % modified by Om Patel
function [F] = loading_plot(p, number, Dataset, ax)

switch nargin
    case 3
        D = categorical(Dataset); % <-- needed for bar plot
        ax = gca; % Use current axes if no axes handle is provided
    case 4
        D = categorical(Dataset); % <-- needed for bar plot
    otherwise
        Dataset = {};
        for i = 1:length(p)
            Dataset = [Dataset, ['Var ', num2str(i)]];
        end
        D = categorical(Dataset); % <-- needed for bar plot
        ax = gca; % Use current axes if no axes handle is provided
end

% Make the actual loadings plot
F = ax; % Return the axes handle
hold(ax, 'on');

% str = '#1BD75F';
str = '#6C6C6D';
color = sscanf(str(2:end), '%2x%2x%2x', [1 3]) / 255;

bar(ax, D, p', 'FaceColor', color, 'EdgeColor', 'black');
grid(ax, 'on');
box(ax, 'on');
ylabel(ax, ['Loadings for Component ' num2str(number)]);

hold(ax, 'off');

end


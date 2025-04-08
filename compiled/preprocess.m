%preprocess data to be transformed and MCS

function[preprocess_data]= preprocess(data)
    %data(:,10) = sqrt(data(:,1)); %sqrts duration
    %data(:,4) = 10.^(data(:,4)/10); %fixes loudness
    %data(:,8) = exp(-data(:,8)); %log speech
    %data(:,9) = exp(-data(:,9)); %log acousticness 
    %data(:,20) = sqrt(-data(:,11)); %log liveness
    % xnew = 10.^(X(:,6)/10);
    % X(:,1) = sqrt(X(:,1)); % duration
    % X(:,9) = sqrt(X(:,9));
    % X(:,4) = log1p(X(:,4));
    % X(:,6) = log1p(-X(:,6)); % loudness
    % X(:, 8) = log1p(X(:, 8)); % speech
    % X(:, 10) = log1p(X(:, 10)); % instrument
    % X(:, 11) = log1p(X(:, 11)); % liveness
    
    average = mean(data);
    st_dev = std(data);
    preprocess_data = (data-average)./st_dev;

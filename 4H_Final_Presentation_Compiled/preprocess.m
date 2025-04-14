%preprocess data to be transformed and MCS

function[preprocess_data]= preprocess(data)
    average = mean(data);
    st_dev = std(data);
    preprocess_data = (data-average)./st_dev;
end
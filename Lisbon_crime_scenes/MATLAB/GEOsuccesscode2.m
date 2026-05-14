
numericData = table2array(CompleteData20241(:, {'mway_dist', 'police_dis'}));

inputdata = numericData(:,1:2);
%outputdata = numericData(:,3);
%Dataset = Dataset(:,8:9);


% Extract the 'success_attack' column from the table
numericData = table2array(CompleteData20241(:, 'success_attack'));

% Convert the extracted column to double format (if not already in double format)
outputData = double(numericData);

% Now 'outputData' can be used as input for your fuzzy inference system
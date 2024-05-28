% Load the saved FIS file
loaded_fis = readfis('Preditor_decision_tree_2.fis');

% Load evaluation dataset (input and target output pairs)
% Example: evaluation_data.mat contains eval_input and eval_target
% Make sure you have loaded inputdata and outputdata from your dataset

% Simulate FIS outputs for evaluation dataset
predicted_output = evalfis(inputdata, loaded_fis);

% Calculate accuracy metrics
mse = mean((predicted_output - outputdata).^2);
mae = mean(abs(predicted_output - outputdata));
r_squared = 1 - sum((outputdata - predicted_output).^2) / sum((outputdata - mean(outputdata)).^2);

disp(['Mean Squared Error (MSE): ', num2str(mse)]);
disp(['Mean Absolute Error (MAE): ', num2str(mae)]);
disp(['R-squared: ', num2str(r_squared)]);

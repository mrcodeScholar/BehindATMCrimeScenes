fis = readfis('3mmadanifinal.fis');
% Define the input values
inputValues = [300,400];
% Evaluate the FIS
outputValues = evalfis(fis, inputValues);
% Display the results
disp(outputValues);

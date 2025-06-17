%%% 1. Initial cleanup, add paths and load in data
clearvars;
close all;
load lab3cardata.mat;

%%% 2. Create helpful variables
mTrain = size(Xtrain, 1);
mTest = size(Xtest, 1);
n = 4;  % Total number of classes
input_layer_size = size(Xtrain, 2);
hidden_layer_size = 10;  % You can experiment with different values
num_labels = n;
lambda = 1;

%%% 3. Randomly initialize parameters
initial_Theta1 = rand(input_layer_size + 1, hidden_layer_size) * 2 - 1;
initial_Theta2 = rand(hidden_layer_size + 1, num_labels) * 2 - 1;
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

%%% 4. Set options for fmincg
options = struct();
options.MaxIter = 100;

%%% 5. Train the neural network
costFunc = @(p) costFunction_NN_reg(p, ...
    input_layer_size, hidden_layer_size, num_labels, ...
    Xtrain, Ytrain, lambda);

[nn_params, ~, ~] = fmincg(costFunc, initial_nn_params, options, false);

%%% 6. Reshape the trained parameters back into Theta1 and Theta2
Theta1 = reshape(nn_params(1:(input_layer_size + 1) * hidden_layer_size), ...
    input_layer_size + 1, hidden_layer_size);

Theta2 = reshape(nn_params((1 + (input_layer_size + 1) * hidden_layer_size):end), ...
    hidden_layer_size + 1, num_labels);

%%% 7. Forward propagation to get predictions
trainScores = forward_propagation(Xtrain, Theta1, Theta2);  % (mTrain x num_labels)
testScores = forward_propagation(Xtest, Theta1, Theta2);    % (mTest x num_labels)

trainPreds = predict_class(trainScores);  % (mTrain x 1)
testPreds = predict_class(testScores);    % (mTest x 1)

%%% 8. Calculate accuracy
trainAccuracy = mean(trainPreds == Ytrain) * 100;
testAccuracy = mean(testPreds == Ytest) * 100;

fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

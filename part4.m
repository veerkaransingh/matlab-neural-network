%%% 1. Initial cleanup, add paths and load in data
%%% DON'T CHANGE
clearvars;
close all;
% addpath('../data');
% addpath('../helper');
load lab3cardata.mat;

%%% 2. Create helpful variables
mTrain = size(Xtrain, 1); % Total number of training examples
mTest = size(Xtest, 1); % Total number of test examples
n = 4; % Total number of classes

%%% 3. Train non-linear SVM classifiers - one vs all using the training data
svmModels = cell(n,1);
for i = 1:n
    svmModels{i} = fitcsvm(Xtrain, Ytrain == i, 'KernelFunction', 'rbf', ...
        'BoxConstraint', 1, 'KernelScale', 'auto');
end

%%% 4. Perform One-Vs-All prediction on the training and test dataset
trainScores = zeros(mTrain, n);
testScores = zeros(mTest, n);

for i = 1:n
    [~, scoreTrain] = predict(svmModels{i}, Xtrain);
    [~, scoreTest] = predict(svmModels{i}, Xtest);
    
    trainScores(:, i) = scoreTrain(:, 2); % Positive class score
    testScores(:, i) = scoreTest(:, 2);
end


%%% 5. Calculate the classification accuracy for the training and test datasets
trainPreds = predict_class(trainScores);
testPreds = predict_class(testScores);

trainAccuracy = mean(trainPreds == Ytrain) * 100;
testAccuracy = mean(testPreds == Ytest) * 100;

fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);
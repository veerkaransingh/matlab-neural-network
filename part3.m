%%% 1. Clear variables and close all figures
clearvars;
close all;

%%% 2. Define input training examples
X = [0 1; 1 1; 1 0; 0 0];
y = [1; 0; 1; 0];

%%% 3. Train a non-linear SVM classifier (using RBF kernel)
SVMModel = fitcsvm(X, y, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 'auto');

%%% 4. Create a grid of values for decision region plotting
[X1, X2] = meshgrid(0:0.01:1);
vals = [X1(:) X2(:)];

%%% 5. Make predictions on the grid (FIXED LINE)
preds = predict(SVMModel, vals);

%%% 6. Plot the decision boundary
figure;
gscatter(vals(:,1), vals(:,2), preds, 'bg');  % now 'preds' is defined
hold on;

%%% 7. Overlay the training points
plot(X(y==1,1), X(y==1,2), 'bo', 'MarkerSize', 12, 'LineWidth', 2);
plot(X(y==0,1), X(y==0,2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);

title('SVM Decision Boundary for XOR');
xlabel('X1');
ylabel('X2');
legend('Class 1 region', 'Class 0 region', 'Class 1 point', 'Class 0 point');
axis tight;
grid on;

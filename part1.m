%%% 1. Clear variables and close all figures
clearvars;
close all;

%%% 2. Define input training examples
X = [0 1; 1 1; 1 0; 0 0];
y = [1; 0; 1; 0];
m = size(X, 1);         % Number of training examples
alpha = 1;              % Learning rate
num_iters = 10000;      % Total training iterations
d0 = 2;                 % Input layer size (2 features)
d1 = 2;                 % Hidden layer size
d2 = 1;                 % Output layer size

%%% 3. Initialize weights with small random values
rng(0);  % For reproducibility
W1 = rand(d0 + 1, d1) * 2 - 1;  % (3 x 2)
W2 = rand(d1 + 1, d2) * 2 - 1;  % (3 x 1)

%%% 4. Stochastic Gradient Descent Loop
for i = 1:num_iters
    for j = 1:m
        % Fetch one training example
        x_j = X(j, :);       % (1 x 2)
        y_j = y(j);          % scalar

        % --- Forward propagation ---
        a1 = [1, x_j];        % Add bias to input (1 x 3)
        z2 = a1 * W1;         % (1 x 2)
        a2 = sigmoid(z2);     % (1 x 2)
        a2 = [1, a2];         % Add bias to hidden layer (1 x 3)
        z3 = a2 * W2;         % (1 x 1)
        a3 = sigmoid(z3);     % output (1 x 1)

        % --- Backpropagation ---
        delta3 = (a3 - y_j) .* dsigmoid(z3);  % (1 x 1)
        delta2 = (delta3 * W2(2:end,:)') .* dsigmoid(z2);  % (1 x 2)

        % --- Gradient update ---
        W2 = W2 - alpha * a2' * delta3;       % (3 x 1)
        W1 = W1 - alpha * a1' * delta2;       % (3 x 2)
    end
end

%%% 5. Plot decision boundary
plot_XOR_and_regions(W1, W2);
title('XOR learned using Neural Network and SGD');

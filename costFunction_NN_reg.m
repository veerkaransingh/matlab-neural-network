function [J, grad] = costFunction_NN_reg(nn_params, ...
    input_layer_size, hidden_layer_size, num_labels, ...
    X, y, lambda)

% Unroll theta parameters from vector
Theta1 = reshape(nn_params(1:(input_layer_size + 1) * hidden_layer_size), ...
    input_layer_size + 1, hidden_layer_size);

Theta2 = reshape(nn_params((1 + (input_layer_size + 1) * hidden_layer_size):end), ...
    hidden_layer_size + 1, num_labels);

m = size(X, 1);
         
% Forward Propagation
a1 = [ones(m, 1) X];                    % (m x d0+1)
z2 = a1 * Theta1;                      
a2 = sigmoid(z2);                      
a2 = [ones(m, 1) a2];                  % (m x d1+1)
z3 = a2 * Theta2;                      
a3 = sigmoid(z3);                      % output (m x num_labels)

% Convert y to one-hot encoding
I = eye(num_labels);
Y = I(y, :);                         % (m x num_labels)

% Cost Function with Regularization
J = (1 / m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));
J = J + (lambda / (2 * m)) * ( ...
    sum(sum(Theta1(2:end,:) .^ 2)) + ...
    sum(sum(Theta2(2:end,:) .^ 2)) );

% Backpropagation
delta3 = a3 - Y;                               % (m x num_labels)
delta2 = (delta3 * Theta2(2:end,:)') .* dsigmoid(z2);  % (m x hidden)

Delta1 = a1' * delta2;                         % (d0+1 x hidden)
Delta2 = a2' * delta3;                         % (hidden+1 x labels)

%  Gradient with Regularization
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

Theta1_grad(2:end,:) = Theta1_grad(2:end,:) + (lambda / m) * Theta1(2:end,:);
Theta2_grad(2:end,:) = Theta2_grad(2:end,:) + (lambda / m) * Theta2(2:end,:);

% Unroll gradients into a vector
grad = [Theta1_grad(:); Theta2_grad(:)];
end

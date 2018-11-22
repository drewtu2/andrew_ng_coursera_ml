function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Prepend with ones for bias
X = [ones(m, 1) X];


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
    function [h0, prediction] = forward_pass(X, Theta1, Theta2)
        % Hypothesis function for a 3 layer neural net
        % input - hidden - output
        % X = num_examples x num_features + 1(bias)
        % Theta1 = units_layer_2 x num_features_in_input + 1(bias)
        % Theta2 = output_units x units_layer_2 + 1(bias)
        a1 = X;
        z2 = a1 * Theta1'; 
        a2 = sigmoid(z2); 
        a2 = [ones(m, 1) a2]; % Prep a2 for calculating a3 by adding bias
        % size(a2) = num_examples x units_layer_2 + 1
        
        z3 = a2 * Theta2'; % size(z3) = num_examples x output_units
       
        h0 = sigmoid(z3);
        [value, indices] = max(h0, [], 2);
        prediction = indices;
    end

    function all_y = map_y(y, num_labels)
        % Returns a (num_examples x num_labels) logical array
        all_y = zeros(size(y, 1), num_labels);
        
        % Each column of all_y becomes a logical array representing whether
        % each row in that column is equal to the index of that column. 
        for label_index=1:num_labels
            all_y(:, label_index) = (y == label_index);
        end
    end

    all_y = map_y(y, num_labels);

    % size(hypothesis) = (num_examples x output_units)
    % size(all_y) = (num_examples x output_units)
    
    pos_term = -1 * all_y .* log(forward_pass(X, Theta1, Theta2)); 
    neg_term = (1 - all_y) .* log(1 - forward_pass(X, Theta1, Theta2));

    J = sum(sum(pos_term - neg_term))/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

    % m = example_number
    % k = class_number
    
    a1 = X;
    z2 = a1 * Theta1'; 
    a2 = sigmoid(z2); 
    a2 = [ones(m, 1) a2]; % Prep a2 for calculating a3 by adding bias
    % size(a2) = num_examples x units_layer_2 + 1

    z3 = a2 * Theta2'; % size(z3) = num_examples x output_units
    a3 = sigmoid(z3);  % a3 = mXk

    all_y = map_y(y, num_labels); % size(all_y) = num_examples x output_unit
    
    delta3 = a3 - all_y;
    
    delta2 = ((Theta2(:, 2:end)'*delta3')');
    delta2 = delta2.*sigmoidGradient(z2);
    
    delta2_accum = delta3'*a2;
    delta1_accum = delta2'*a1;
    
    Theta2_grad = delta2_accum/m;
    Theta1_grad = delta1_accum/m;
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

stripped_t1 = Theta1(:, 2:end);
stripped_t2 = Theta2(:, 2:end);

regularization = lambda/(2*m) * ...
    (sum(sum(stripped_t1.^2)) + sum(sum(stripped_t2.^2)));

J = J + regularization;

grad1_reg = (lambda/m)*stripped_t1;
grad2_reg = (lambda/m)*stripped_t2;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + grad1_reg;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + grad2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

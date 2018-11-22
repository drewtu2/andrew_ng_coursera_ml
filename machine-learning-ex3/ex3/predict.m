function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add a 1 into X(0)
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% X (Nx401)
% Theta1 (25x401)
% Theta2 (10x26)

% Find the values of each weight times the corresponding input
z2 = X*Theta1';
% Apply the activation function to find the result of the first hidden
% layer
a2 = sigmoid(z2);
% Add the bias into a1 to prep for use as input into second hidden layer
a2 = [ones(m, 1) a2];

% Find the values of each weight times the corresponding input
z3 = a2*Theta2';
% Apply the actiation function to find the result of the second hidden
% layer
a3 = sigmoid(z3);

[v, p] = max(a3, [], 2);
% =========================================================================


end

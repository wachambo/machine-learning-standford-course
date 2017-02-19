function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Tip:
%
% You can get a one-line function for sigmoid(z) if you use only element-wise
% operators.
%
%     - The exp() function is element-wise.
%     - The addition operator is element-wise.
%     - Use the element-wise division operator ./
%
% Combine these elements with a few parenthesis, and operate only on the
% parameter 'z'. The return value 'g' will then be the same size as 'z',
% regardless of what data 'z' contains.

g = 1./(1 + exp(-z));
% =============================================================

end

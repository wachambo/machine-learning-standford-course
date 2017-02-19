function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Compute the cost and gradient of regularized linear regression for a
% particular choice of theta.
%
% You should set J to the cost and grad to the gradient.
% }}}

% Tips:
% {{{
% You only need the first three steps of the gradientDescent() tutorial, plus
% scaling by 1/m. That's gives us the gradient. Since we let fmincg() perform
% gradient descent for us, we just have to compute the cost and gradient. We
% don't use a for-loop over the number of iterations, or use any learning rate.
% The fmincg() function does that for us.
%
% So now you've got unregularized cost J, and unregularized gradient 'grad'.
%
% For the cost regularization:
%
% Set theta(1) to 0.  Compute the sum of all of the theta values squared.
% One handy way to do this is sum(theta.^2). Since theta(1) has been forced
% to zero, it doesn't add to the regularization term.  Now scale this value
% by lambda / (2*m), and add it to the unregularized cost.
%
% For the gradient regularization:
%
% The regularized gradient term is theta scaled by (lambda / m). Again,
% since theta(1) has been set to zero, it does not contribute to the
% regularization term.  Add this vector to the unregularized portion.
% }}}

h        = X * theta;
sum1     = (h - y).^2;
theta(1) = 0;
J        = (1/(2*m)) * sum(sum1) + (lambda/(2*m)) * sum(theta.^2);

err  = h - y;
grad = (1/m) * (X' * err) + (lambda/m) * theta;
% =========================================================================

grad = grad(:);

end;


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

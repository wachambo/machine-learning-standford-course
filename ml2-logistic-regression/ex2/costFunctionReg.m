function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Compute the cost of a particular choice of theta.
% You should set J to the cost.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta
% }}}


% Tips for cost:
% {{{
% 1. The hypothesis is a vector, formed from the sigmoid() of the products of X
%    and θ. See the equation on ex2.pdf - Page 4. Be sure your sigmoid()
%    function passes the submit grader before going any further.
%
% 2. First focus on the portions of the cost equation. There are two
%    vector terms that are subtracted from each other, then are summed and
%    scaled. Alternatively we can sum each vector individually. That's a
%    more efficient solution, so we'll proceed with that method.
%
% 3. The sum of -y multiplied by the natural log of h.
%    Note that the natural log function is log(). Don't use log10(). Since we
%    want the sum of the products, we can use a vector multiplication. The size
%    of each argument is (m x 1), and we want the vector product to be a scalar,
%    so use a transposition so that (1 x m) times (m x 1) gives a result
%    of (1 x 1), a scalar.
%
% 4. Subtract the right-side term from the left-side term
%
% 5. Scale the result by 1/m. This is the unregularized cost.
%
% 6. Now we have only the regularization term remaining. We want the
%    regularization to exclude the bias feature, so we can set theta(1) to zero.
%    Since we already calculated h, and theta is a local variable, we can modify
%    theta(1) without causing any problems.
%
% 7. Now we need to calculate the sum of the squares of theta. Since we've set
%    theta(1) to zero, we can square the entire theta vector. If we
%    vector-multiply theta by itself, we will calculate the sum automatically.
%    So use the same method we used in Steps 3 and 4 to multiply theta by
%    itself with a transposition.
%
% 8. Now scale the cost regularization term by (lambda / (2 * m)). Be sure you
%    use enough sets of parenthesis to get the correct result.
%
% 0. Now add your unregularized and regularized cost terms together
% }}}

% Tips for gradient:
% {{{
% 1. Recall that the hypothesis vector h is the
%    sigmoid() of the product of X and θ (see ex2.pdf - Page 4). You probably
%    already calculated h for the cost J calculation.
%
% 2. The left-side term is the vector product of X and (h - y), scaled by
%    1/m. You'll need to transpose and swap the product terms so the result is
%    (m x n)' times (m x 1) giving you a (n x 1) result. This is the
%    unregularized gradient. Note that the vector product also includes the
%    required summation.
%
% 3. Then set theta(1) to 0 (if you haven't already).
%
% 4. Then calculate the regularized gradient
%    term as theta scaled by (lambda / m).
%
% 5. The grad value is the sum of
%    the Step 2 and Step 4 results. Since you forced theta(1) to be zero, the
%    grad(1) term will only be the unregularized value.
% }}}


h        = sigmoid(X * theta);
sum1     = -y' * log(h);
sum2     = (1 - y)' * log(1 - h);
sub      = sum1 - sum2;
theta(1) = 0;
J        = (1/m) * sum(sub) + (lambda/(2*m)) * (theta' * theta);

err      = h - y;
grad     = (1/m) * (X' * err) + (lambda/m) * theta;
% =============================================================

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

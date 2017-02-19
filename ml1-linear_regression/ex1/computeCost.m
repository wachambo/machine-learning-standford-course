function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Compute the cost of a particular choice of theta
% You should set J to the cost.
% }}}

% Tips:
% {{{
% The first line of code will compute a vector 'h' containing all of the
% hypothesis values - one for each training example (i.e. for each row of X).
%
% The hypothesis (also called the prediction) is simply the product of X and
% theta. So your first line of code is...
%     h = {multiply X and theta, in the
%          proper order that the inner dimensions match}
%
% Since X is size (m x n) and theta is size (n x 1), you arrange the order of
% operators so the result is size (m x 1).
%
% The second line of code will compute the difference between the hypothesis and
% y - that's the error for each training example. Difference means subtract.
%     error = {the difference between h and y}
%
% The third line of code will compute the square of each of those error terms
% (using element-wise exponentiation),
%
% An example of using element-wise exponentiation - try this in your workspace
% command line so you see how it works.
%     v = [-2 3]
%     v_sqr = v.^2
%
% So, now you should compute the squares of the error terms:
%     error_sqr = {use what you have learned}
%
% Next, here's an example of how the sum function works (try this from your
% command line)
%     q = sum([1 2 3])
%
% Now, we'll finish the last two steps all in one line of code. You need to
% compute the sum of the error_sqr vector, and scale the result (multiply) by
% 1/(2*m). That completed sum is the cost value J.
%     J = {multiply 1/(2*m) times the sum of the error_sqr vector}
% }}}

h   = X * theta;
err = h - y;
J   = 1/(2*m) * sum(err.^2);
% =========================================================================

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

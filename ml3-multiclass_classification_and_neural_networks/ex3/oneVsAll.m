function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% You should complete the following code to train num_labels logistic regression
% classifiers with regularization parameter lambda.
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
% }}}

% Tips:
% {{{
% all_theta is a matrix, where there is a row for each of the trained thetas. In
% the exercise example, there are 10 rows, of 401 elements each. You know this
% because that's how all_theta was initialized in line 15 of the script
% template.
%
% (note that the submit grader's test case doesn't have 401 elements or 10 rows
% - your function must work for any size data set - so use the "num_labels"
% variable).
%
% Each call to fmincg() returns a theta vector. Be sure you use the lambda value
% provided in the function header.
%
% You then need to copy that vector into a row of all_theta.
%
% The oneVsAll.m script template contains several Hints and a code example to
% guide your work.
%
% Type these commands in your workspace to see how to copy a vector into a
% matrix:
%     Q = zeros(5,3)      % create a test matrix of all-zeros
%     v = [1 2 3]'        % create a column vector
%     Q(2,:) = v          % copy v into the 2nd row of Q
%
% The syntax "(2,:)" means "use all columns of the 2nd row".
% }}}

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);  % options for fmincg

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost
for c = 1:num_labels,
  [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
  all_theta(c, :) = theta';
end
% =========================================================================


end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

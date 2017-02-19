function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% First, for each feature dimension, compute the mean of the feature and
% subtract it from the dataset, storing the mean value in mu. Next, compute the
% standard deviation of each feature and divide each feature by it's standard
% deviation, storing the standard deviation in sigma.
%
% Note that X is a matrix where each column is a feature and each row is an
% example. You need to perform the normalization separately for each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
% }}}

% Tips:
% {{{
% There are a couple of methods to accomplish this. The method here is one I use
% that doesn't rely on automatic broadcasting or the bsxfun() or repmat()
% functions.
%
%     You can use the mean() and sigma() functions to get the mean and std
%     deviation for each column of X. These are returned as row vectors (1 x n)
%     Now you want to apply those values to each element in every row of the X
%     matrix. One way to do this is to duplicate these vectors for each row in
%     X, so they're the same size.
%
% One method to do this is to create a column vector of all-ones - size (m x 1)
% - and multiply it by the mu or sigma row vector (1 x n). Dimensionally, (m x
% 1) * (1 x n) gives you a (m x n) matrix, and every row of the resulting matrix
% will be identical.
%
%     Now that X, mu, and sigma are all the same size, you can use element-wise
%     operators to compute X_normalized.
%
% Try these commands in your workspace:
%     X = [1 2 3; 4 5 6]        % creates a test matrix
%     mu = mean(X)              % returns a row vector
%     sigma = std(X)            % returns a row vector
%     m = size(X, 1)            % returns the number of rows in X
%     mu_matrix = ones(m, 1) * mu
%     sigma_matrix = ones(m, 1) * sigma
%
% Now you can subtract the mu matrix from X, and divide element-wise by the
% sigma matrix, and arrive at X_normalized.
%
% You can do this even easier if you're using a Matlab or Octave version that
% supports automatic broadcasting - then you can skip the "multiply by a column
% of 1's" part.
%
% You can also use the bsxfun() or repmat() functions. Be advised the bsxfun()
% has a non-obvious syntax that I can never remember, and repmat() runs rather
% slowly.
% }}}




% ============================================================

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

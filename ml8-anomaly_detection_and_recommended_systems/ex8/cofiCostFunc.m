function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Compute the cost function and gradient for collaborative filtering.
% Concretely, you should first implement the cost function (without
% regularization) and make sure it is matches our costs. After that, you should
% implement the gradient and use the checkCostFunction routine to check that the
% gradient is correct. Finally, you should implement regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
% }}}

% Tips:
% {{{
% Definitions:
%
% R: a matrix of observations (binary values). Dimensions are (movies x users)
%
% Y: a matrix of movie ratings: Dimensions are (movies x users)
%
% X: a matrix of movie features (0 to 5): Dimensions are (movies x features)
%
% Theta: a matrix of feature weights: Dimensions are (users x features)
%
% - Compute the predicted movie ratings for all users using the product of X and
% Theta. A transposition may be needed.
%
% Dimensions of the result should be (movies x users).
%
% - Compute the movie rating error by subtracting Y from the predicted
% ratings.
%
% - Compute the "error_factor" my multiplying the movie rating error by the R
% matrix. The error factor will be 0 for movies that a user has not rated. Use
% the type of multiplication by R (vector or element-wise) so the size of the
% error factor matrix remains unchanged (movies x users).
%
% (Note: there is a quirk in the submit grader's test case that requires you to
% use the R matrix to ignore movies that have had no ratings).
%
% Calculate the cost:
%
% - Using the formula on Page 9 of ex8.pdf, compute the unregularized cost as a
% scaled sum of the squares of all of the terms in error_factor. The result
% should be a scalar.
%
% - Test your code using ex8_cofi.m and the additional test cases. You should
% get a passing grade for this portion from the submit script.
%
% Calculate the gradients (ref: the formulas on Page 10 of ex8,pdf):
%
% - The X gradient is the product of the error factor and the Theta matrix. The
% sum is computed automatically by the vector multiplication. Dimensions are
% (movies x features)
%
% - The Theta gradient is the product of the error factor and the X matrix. A
% transposition may be needed. The sum is computed automatically by the vector
% multiplication. Dimensions are (users x features)
%
% - Test your code, then submit this portion.
%
% Calculate the regularized cost:
%
% - Using the formula on the top of Page 13 of ex8.pdf, compute the
% regularization term as the scaled sum of the squares of all terms in Theta and
% X. The result should be a scalar. Note that for Recommender Systems there are
% no bias terms, so regularization should include all columns of X and Theta.
%
% - Add the regularized and un-regularized cost terms.
%
% - Test your code, then submit this portion.
%
% Calculate the gradient regularization terms (ref: the formulas in the middle
% of Page 13 of ex8.pdf)
%
% - The X gradient regularization is the X matrix scaled by lambda.
%
% - The Theta gradient regularization is the Theta matrix scaled by lambda.
%
% - Add the regularization terms to their unregularized values.
%
% - Test your code, then submit this portion
% }}}

err         = ((X * Theta' - Y) .* R);
squared_err = err .^ 2;
J           = ((1/2) * sum(squared_err(:))) + ((lambda/2) * sum(Theta(:) .^ 2)) + ((lambda/2) * sum(X(:) .^ 2));

X_grad     = err * Theta .+ (lambda .* X);
Theta_grad = err' *  X .+ (lambda .* Theta);
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

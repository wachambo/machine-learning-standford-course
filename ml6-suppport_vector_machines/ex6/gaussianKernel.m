function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Fill in this function to return the similarity between x1 and x2 computed
% using a Gaussian kernel with bandwidth sigma
% }}}

% Tip:
% {{{
% The method is similar to that used for sigmoid.m in ex2. The numerator is
% the sum of the squares of the difference of two vectors. That's just like
% computing the linear regression cost function. Use exp() and scale by the
% value given in the formula (top of page 6 of ex6.pdf)
% }}}

diff = x1 - x2;
% norm = sum(diff .^ 2); % norm or magnitude
norm = diff' * diff; % norm or magnitude
sim  = exp(-norm / (2 * sigma ^ 2));
% =============================================================

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :<Paste>

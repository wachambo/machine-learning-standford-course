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

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% You should complete the code by working through the following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial
%         derivatives of the cost function with respect to Theta1 and Theta2 in
%         Theta1_grad and Theta2_grad, respectively. After implementing Part 2,
%         you can check that your implementation is correct by running
%         checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% }}}

% Glossary:
% {{{
% Each of these variables will have a subscript, noting which NN layer it is
% associated with.
%
% Θ: A Theta matrix of weights to compute the inner values of the neural
% network. When we used a vector theta, it was noted with the lower-case theta
% character θ.
%
% z : is the result of multiplying a data vector with a Θ matrix. A typical
% variable name would be "z2".
%
% a : The "activation" output from a neural layer. This is always generated
% using a sigmoid function g() on a z value. A typical variable name would be
% "a2".
%
% δ : lower-case delta is used for the "error" term in each layer. A typical
% variable name would be "d2".
%
% Δ : upper-case delta is used to hold the sum of the product of a δ value with
% the previous layer's a value. In the vectorized solution, these sums are
% calculated automatically though the magic of matrix algebra. A typical
% variable name would be "Delta2".
%
% Θ_gradient : This is the thing we're solving for, the partial derivative of
% theta. There is one of these variables associated with each Δ. These values
% are returned by nnCostFunction(), so the variable names must be "Theta1_grad"
% and "Theta2_grad".
%
% g() is the sigmoid function.
%
% g′() is the sigmoid gradient function.
%
% Tip: One handy method for excluding a column of bias units is to use the
% notation SomeMatrix(:,2:end). This selects all of the rows of a matrix, and
% omits the entire first column.
%
% See the Appendix at the bottom of the tutorial for information on the sizes of
% the data objects.
%
% A note regarding bias units, regularization, and back-propagation:
%
% There are two methods for handing exclusion of the bias units in the Theta
% matrices in the back-propagation and gradient calculations. I've described
% only one of them here, it's the one that I understood the best. Both methods
% work, choose the one that makes sense to you and avoids dimension errors. It
% matters not a whit whether the bias unit is excluded before or after it is
% calculated - both methods give the same results, though the order of
% operations and transpositions required may be different. Those with contrary
% opinions are welcome to write their own tutorial.
% }}}

% Tips for Forward Propagation:
% {{{
% We'll start by outlining the forward propagation process. Though this was
% already accomplished once during Exercise 3, you'll need to duplicate some of
% that work because computing the gradients requires some of the intermediate
% results from forward propagation. Also, the y values in ex4 are a matrix,
% instead of a vector. This changes the method for computing the cost J.
%
% 1. Expand the 'y' output values into a matrix of single values (see ex4.pdf
%    Page 5). This is most easily done using an eye() matrix of size num_labels,
%    with vectorized indexing by 'y'. A useful variable name would be "y_matrix",
%    as this:
%    y_matrix = eye(num_labels)(y,:)
%
%    Note: For MATLAB users, this expression must be split into two lines, such
%    as...
%        eye_matrix = eye(num_labels)
%        y_matrix = eye_matrix(y,:)
%
%    Discussions of other methods are available in the Course Wiki - Programming
%    Exercises section.
%
% 2. Perform the forward propagation:
%
%    a1 equals the X input matrix with a column of 1's added (bias units) as the
%    first column.
%
%    z2 equals the product of a1 and Θ1
%
%    a2 is the result of passing z2 through g()
%
%    Then add a column of bias units to a2 (as the first column).
%
%    NOTE: Be sure you DON'T add the bias units as a new row of Theta.
%
%    z3 equals the product of a2 and Θ2
%
%    a3 is the result of passing z3 through g()
% }}}

% Tips for Cost Function, non-regularized:
% {{{
% 3. Compute the unregularized cost according to ex4.pdf (top of Page 5), using
%    a3, your y_matrix, and m (the number of training examples). Note that the
%    'h' argument inside the log() function is exactly a3. Cost should be a
%    scalar value. Since y_matrix and a3 are both matrices, you need to compute
%    the double-sum.
%
%    Remember to use element-wise multiplication with the log() function. Also,
%    we're using the natural log, not log10().
%
%    Now you can run ex4.m to check the unregularized cost is correct, then you
%    can submit this portion to the grader.
% }}}

% Tips for Cost Regularization:
% {{{
% 4. Compute the regularized component of the cost according to ex4.pdf Page 6,
%    using Θ1 and Θ2 (excluding the Theta columns for the bias units), along
%    with λ, and m. The easiest method to do this is to compute the
%    regularization terms separately, then add them to the unregularized cost
%    from Step 3.
% }}}

% Tips for backpropagation:
% {{{
% You can design your code for backpropagation based on analysis of the
% dimensions of all of the data objects. This tutorial uses the vectorized
% method, for easy comprehension and speed of execution.
%
% Reference the four steps outlined on Page 9 of ex4.pdf.
%
% ---------------------------------
%
% Let:
%
% m = the number of training examples
%
% n = the number of training features, including the initial bias unit.
%
% h = the number of units in the hidden layer - NOT including the bias unit
%
% r = the number of output classifications
%
% -------------------------------
%
% 1. Perform forward propagation, see the separate tutorial if necessary.
%
% 2. δ3 or d3 is the difference between a3 and the y_matrix. The dimensions are
%    the same as both, (m x r).
%
% 3. z2 came from the forward propagation process - it's the product of a1 and
%    Theta1, prior to applying the sigmoid() function.
%    Dimensions are (m x n) ⋅ (n x h) --> (m x h)
%
% 4: δ2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the
%    product of d3 and Theta2(no bias), then element-wise scaled by sigmoid
%    gradient of z2. The size is (m x r) ⋅ (r x h) --> (m x h). The size is the
%    same as z2, as must be.
%
% 5: Δ1 or Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m x n) -->
%    (h x n)
%
% 6: Δ2 or Delta2 is the product of d3 and a2. The size is (r x m) ⋅ (m x [h+1])
%    --> (r x [h+1])
%
% 7: Theta1_grad and Theta2_grad are the same size as their respective Deltas,
%    just scaled by 1/m
%
% Now you have the unregularized gradients. Check your results using ex4.m, and
% submit this portion to the grader.
% }}}

% Tips for Regularization of the gradient
% {{{
% Since Theta1 and Theta2 are local copies, and we've already computed our
% hypothesis value during forward-propagation, we're free to modify them to make
% the gradient regularization easy to compute.
%
% 8. So, set the first column of Theta1 and Theta2 to all-zeros. Here's a method
%    you can try in your workspace console:
%        Q = rand(3,4)       % create a test matrix
%        Q(:,1) = 0          % set the 1st column of all rows to 0
%
% 9. Scale each Theta matrix by λ/m. Use enough parenthesis so the operation is
%    correct.
%
% 10.Add each of these modified-and-scaled Theta matrices to the un-regularized
%    Theta gradients that you computed earlier.
% }}}


% Redecode labels as vectors: each y(i) value as y(i)-th row from eye matrix
y_matrix = eye(num_labels)(y, :);

% Feedforward
a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
m2 = size(z2, 1); % number of columns of z2
a2 = [ones(m2, 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Cost
h    = a3;
sum1 = -y_matrix .* log(h);
sum2 = (1 .- y_matrix) .* log(1 - h);
sub  = sum1 - sum2;
J    = (1 / m) * sum(sum(sub));

% Regularized cost
sum1    = sum(sum(Theta1(:, 2:end) .^ 2));
sum2    = sum(sum(Theta2(:, 2:end) .^ 2));
penalty = (lambda/(2*m)) * (sum1 + sum2);
J       = J + penalty;

% -------------------------------------------------------------
% Backpropragation
% Delta1 = zeros(size(Theta1));
% Delta2 = zeros(size(Theta2));
%
% for t = 1:m  % training examples
%     % feedforward
%     a1 = X(t, :)'; % t-th row of X
%     a1 = [1; a1];  % include bias unit
%     z2 = Theta1 * a1;
%     a2 = sigmoid(z2);
%     a2 = [1; a2];  % include bias unit
%     z3 = Theta2 * a2;
%     a3 = sigmoid(z3);
%
%     % deltaL errors in layer L
%     d3 = a3 - y_matrix(t, :)';
%     d2 = Theta2' * d3;
%     d2 = d2(2:end, :);  % skip d2(0)
%     d2 = d2 .* sigmoidGradient(z2);
%
%     Delta1 = Delta1 + d2 * a1';
%     Delta2 = Delta2 + d3 * a2';
% end

d3     = a3 - y_matrix;
d2     = d3 * Theta2(:, 2:end);
d2     = sigmoidGradient(z2) .* d2;
Delta1 = d2' * a1;
Delta2 = d3' * a2;

% Gradient
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% -------------------------------------------------------------
% Regularized gradient
temp1       = Theta1;
temp2       = Theta2;
temp1(:, 1) = 0;
temp2(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda/m) * temp1;
Theta2_grad = Theta2_grad + (lambda/m) * temp2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

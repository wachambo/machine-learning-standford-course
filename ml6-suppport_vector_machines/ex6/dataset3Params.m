function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;
C = 0;
sigma = 0

% ====================== YOUR CODE HERE ======================
% Instructions:
% {{{
% Fill in this function to return the optimal C and sigma learning parameters
% found using the cross validation set.  You can use svmPredict to predict the
% labels on the cross validation set. For example, predictions =
% svmPredict(model, Xval); will return the predictions on the cross validation
% set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
% }}}

% Tip:
% {{{
% One method is to use two nested for-loops - each one iterating over the
% range of C or sigma values given in the ex6.pdf file.
%
% Inside the inner loop:
%   - Train the model using svmTrain with X, y, a value for C, and the
%     gaussian kernel using a value for sigma.
%   - Compure the predictions for the validation set using svmPredict()
%     with model and Xval.
%   - Compute the error between your predictions and yval.
%   - When you find a new minimum error, save the C and sigma values that
%     were used.
% }}}

values = [0.01 0.03 0.1 0.3 1 3 10 30]; % possible values for C and sigma
err_min = inf; % minimum error

for C_test = values
    for sigma_test = values
        kernelFun   = @(x1, x2) gaussianKernel(x1, x2, sigma_test);
        model       = svmTrain(X, y, C_test, kernelFun);
        predictions = svmPredict(model, Xval);
        err         = mean(double(predictions ~= yval));
        printf('error = %.2f', err)
        if err <= err_min
            err_min = err;
            C       = C_test;
            sigma   = sigma_test;
            fprintf('minimum found [C sigma] = [%.2f %.2f]\n', C, sigma);
        end
    end
end
% =========================================================================

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

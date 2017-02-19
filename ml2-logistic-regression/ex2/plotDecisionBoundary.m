function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
%   positive examples and o for the negative examples. X is assumed to be
%   a either
%   1) Mx3 matrix, where the first column is an all-ones column for the
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Tips:
% {{{
% For logistic regression, h = sigmoid(X * theta). This describes the
% relationship between X, theta, and h.
%
% We know theta (from gradient descent).
%
% We know h - by definition, the decision boundary is the locus of points where
% h = 0.5, or equivalently (X * theta) = 0, since the sigmoid(0) is 0.5.
%
% Now we can write out the equation for the case where we have two features and
% a bias unit, and we write X as [x0x1x2] and theta as [θ0θ1θ2]
%
% 0=x0θ0+x1θ1+x2θ2
%
% x0 is the bias unit, it is hard-coded to 1.
%
% 0=θ0+x1θ1+x2θ2
%
% Solve for x2
%
% x2=−(θ0+x1θ1)/θ2
%
% Now, to draw a line, you need two points. So pick two values for x1 - anything
% near the minimum and maximum of the training set will serve. Compute the
% corresponding values for x2, and plot the (x1x2) pairs on the horizontal and
% vertical axes, then draw a line through them.
%
% This line represents the decision boundary.
%
% This is exactly what the plotDecisionBoundary() function does. x2 is the
% variable "plot_y", and x1 is the variable "plot_x".
% }}}

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)

    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end


% vim: set ai ts=4 sw=4 sts=4 tw=78 et ft=matlab fdm=marker fen :

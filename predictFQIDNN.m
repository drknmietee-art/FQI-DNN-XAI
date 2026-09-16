function [y_pred, y_scores] = predictFQIDNN(net, X_test)
% PREDICTFQIDNN Make predictions using trained FQI-DNN network
%
%   Inputs:
%     net    - trained network from buildAndTrainFQIDNN
%     X_test - [N x d] test features
%
%   Outputs:
%     y_pred   - [N x 1] predicted binary labels (0 or 1)
%     y_scores - [N x 1] predicted probability of class 1

    % Predict
    y_cat = classify(net, X_test);
    y_pred = double(y_cat) - 1; % Convert from categorical (1,2) to (0,1)

    % Get probability scores
    scores = predict(net, X_test);
    if size(scores, 2) >= 2
        y_scores = scores(:, 2); % Probability of class 1
    else
        y_scores = scores(:, 1);
    end
end

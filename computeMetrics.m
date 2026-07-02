function [acc, sen, spe, pre, f1, aucVal] = computeMetrics(y_true, y_pred, y_scores)
% COMPUTEMETRICS Compute classification performance metrics (Eqs. 21-24)
%
%   Inputs:
%     y_true   - [N x 1] true labels (0 or 1)
%     y_pred   - [N x 1] predicted labels (0 or 1)
%     y_scores - [N x 1] predicted probabilities for class 1
%
%   Outputs:
%     acc - accuracy (Eq. 21)
%     sen - sensitivity/recall (Eq. 22)
%     spe - specificity (Eq. 23)
%     pre - precision
%     f1  - F1-score (Eq. 24)
%     aucVal - AUC-ROC

    % Confusion matrix elements
    TP = sum(y_pred == 1 & y_true == 1);
    TN = sum(y_pred == 0 & y_true == 0);
    FP = sum(y_pred == 1 & y_true == 0);
    FN = sum(y_pred == 0 & y_true == 1);

    % Accuracy (Eq. 21)
    acc = (TP + TN) / (TP + TN + FP + FN + eps);

    % Sensitivity / Recall (Eq. 22)
    sen = TP / (TP + FN + eps);

    % Specificity (Eq. 23)
    spe = TN / (TN + FP + eps);

    % Precision
    pre = TP / (TP + FP + eps);

    % F1-Score (Eq. 24)
    f1 = 2 * (pre * sen) / (pre + sen + eps);

    % AUC-ROC
    if length(unique(y_true)) > 1 && ~isempty(y_scores)
        [~, ~, ~, aucVal] = perfcurve(y_true, y_scores, 1);
    else
        aucVal = 0.5;
    end
end

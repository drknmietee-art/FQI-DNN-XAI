function baselineResults = runBaselineComparisons(X, y, cv)
% RUNBASELINECOMPARISONS Run all baseline classifiers for comparison
%
%   Implements Table 4 comparisons: LR, SVM, RF, XGBoost (TreeBagger),
%   standard DNN, and stores results.
%
%   Inputs:
%     X  - [N x d] normalized features
%     y  - [N x 1] binary labels
%     cv - cvpartition object for 10-fold CV
%
%   Output:
%     baselineResults - struct with results for each baseline

    K = cv.NumTestSets;
    methods = {'LogisticRegression', 'SVM_RBF', 'RandomForest', ...
               'GradientBoosting', 'KNN', 'NaiveBayes'};
    nMethods = length(methods);

    baselineResults = struct();

    for m = 1:nMethods
        methodName = methods{m};
        acc = zeros(K, 1);
        sen = zeros(K, 1);
        spe = zeros(K, 1);
        pre = zeros(K, 1);
        f1  = zeros(K, 1);
        aucVals = zeros(K, 1);
        allTrue = [];
        allScores = [];

        for fold = 1:K
            trainIdx = training(cv, fold);
            testIdx = test(cv, fold);

            X_tr = X(trainIdx, :);
            y_tr = y(trainIdx);
            X_te = X(testIdx, :);
            y_te = y(testIdx);

            switch methodName
                case 'LogisticRegression'
                    mdl = fitglm(X_tr, y_tr, 'Distribution', 'binomial');
                    y_scores = predict(mdl, X_te);
                    y_pred = double(y_scores >= 0.5);

                case 'SVM_RBF'
                    mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'rbf', ...
                        'KernelScale', 'auto', 'Standardize', true, ...
                        'BoxConstraint', 1);
                    mdl = fitPosterior(mdl);
                    [y_pred, scores] = predict(mdl, X_te);
                    y_scores = scores(:, 2);

                case 'RandomForest'
                    mdl = TreeBagger(200, X_tr, y_tr, 'Method', 'classification', ...
                        'OOBPrediction', 'on', 'MinLeafSize', 5);
                    [y_pred_cell, scores] = predict(mdl, X_te);
                    y_pred = str2double(y_pred_cell);
                    y_scores = scores(:, 2);

                case 'GradientBoosting'
                    % Ensemble of boosted trees (XGBoost-like)
                    mdl = fitcensemble(X_tr, y_tr, 'Method', 'AdaBoostM1', ...
                        'NumLearningCycles', 200, ...
                        'Learners', templateTree('MaxNumSplits', 10));
                    [y_pred, scores] = predict(mdl, X_te);
                    y_scores = scores(:, 2);

                case 'KNN'
                    mdl = fitcknn(X_tr, y_tr, 'NumNeighbors', 7, ...
                        'Distance', 'minkowski', 'Standardize', true);
                    [y_pred, scores] = predict(mdl, X_te);
                    y_scores = scores(:, 2);

                case 'NaiveBayes'
                    mdl = fitcnb(X_tr, y_tr, 'DistributionNames', 'kernel');
                    [y_pred, scores] = predict(mdl, X_te);
                    y_scores = scores(:, 2);
            end

            [acc(fold), sen(fold), spe(fold), pre(fold), f1(fold), aucVals(fold)] = ...
                computeMetrics(y_te, y_pred, y_scores);

            allTrue = [allTrue; y_te];
            allScores = [allScores; y_scores];
        end

        baselineResults.(methodName).accuracy = acc;
        baselineResults.(methodName).sensitivity = sen;
        baselineResults.(methodName).specificity = spe;
        baselineResults.(methodName).precision = pre;
        baselineResults.(methodName).f1score = f1;
        baselineResults.(methodName).auc = aucVals;
        baselineResults.(methodName).allTrue = allTrue;
        baselineResults.(methodName).allScores = allScores;

        fprintf('  %s: Acc=%.2f ± %.2f%%\n', methodName, ...
                mean(acc)*100, std(acc)*100);
    end
end

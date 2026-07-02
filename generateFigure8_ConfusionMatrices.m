function generateFigure8_ConfusionMatrices(results, baselineResults, outputDir)
% FIGURE 8: Confusion matrices
%   Subfigures: (a) Proposed, (b-d) Top baselines

    fig = figure('Position', [100 100 1000 250], 'Color', 'w');

    % (a) Proposed
    subplot(1, 4, 1);
    cm = confusionmat(results.allTrue, results.allPred);
    plotConfMat(cm, '(a) FQI-DNN-XAI');

    % (b-d) Baselines
    topBaselines = {'GradientBoosting', 'RandomForest', 'SVM_RBF'};
    subLabels = {'(b)', '(c)', '(d)'};
    for m = 1:3
        subplot(1, 4, m + 1);
        if isfield(baselineResults, topBaselines{m})
            br = baselineResults.(topBaselines{m});
            y_pred_bl = double(br.allScores >= 0.5);
            cm_bl = confusionmat(br.allTrue, y_pred_bl);
            plotConfMat(cm_bl, sprintf('%s %s', subLabels{m}, topBaselines{m}));
        end
    end

    sgtitle('Figure 8. Confusion Matrices', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure8_ConfusionMatrices.fig'));
    saveas(fig, fullfile(outputDir, 'Figure8_ConfusionMatrices.png'));
    fprintf('  Figure 8: Confusion matrices saved.\n');
    close(fig);
end

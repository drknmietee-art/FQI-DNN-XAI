function generateFigure7_ROCCurves(results, baselineResults, outputDir)
% FIGURE 7: ROC curves comparing all methods
%   Subfigures: (a) All methods, (b) Zoomed top-5

    fig = figure('Position', [100 100 1000 450], 'Color', 'w');
    colors = lines(8);

    methodNames = fieldnames(baselineResults);

    % (a) All methods
    subplot(1, 2, 1);
    hold on;
    for m = 1:length(methodNames)
        br = baselineResults.(methodNames{m});
        [fpr, tpr, ~, ~] = perfcurve(br.allTrue, br.allScores, 1);
        plot(fpr, tpr, 'Color', colors(m, :), 'LineWidth', 1.2);
    end
    % Proposed method
    [fpr_p, tpr_p, ~, ~] = perfcurve(results.allTrue, results.allScores, 1);
    plot(fpr_p, tpr_p, 'r-', 'LineWidth', 2.5);
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.8);

    legend([methodNames; {'FQI-DNN-XAI (Proposed)'; 'Random'}], ...
           'Location', 'southeast', 'FontSize', 7);
    xlabel('False Positive Rate', 'FontSize', 11);
    ylabel('True Positive Rate', 'FontSize', 11);
    title('(a) ROC Curves - All Methods', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 10);

    % (b) Zoomed view (top-left corner)
    subplot(1, 2, 2);
    hold on;
    % Plot top methods
    topMethods = {'GradientBoosting', 'RandomForest', 'SVM_RBF'};
    for m = 1:length(topMethods)
        if isfield(baselineResults, topMethods{m})
            br = baselineResults.(topMethods{m});
            [fpr, tpr, ~, ~] = perfcurve(br.allTrue, br.allScores, 1);
            plot(fpr, tpr, 'Color', colors(m+3, :), 'LineWidth', 1.5);
        end
    end
    plot(fpr_p, tpr_p, 'r-', 'LineWidth', 2.5);
    xlim([0 0.3]); ylim([0.7 1.0]);
    legend([topMethods, {'FQI-DNN-XAI'}], 'Location', 'southeast', 'FontSize', 9);
    xlabel('False Positive Rate', 'FontSize', 11);
    ylabel('True Positive Rate', 'FontSize', 11);
    title('(b) Zoomed View - Top Methods', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 10);

    sgtitle('Figure 7. ROC Curves', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure7_ROCCurves.fig'));
    saveas(fig, fullfile(outputDir, 'Figure7_ROCCurves.png'));
    fprintf('  Figure 7: ROC curves saved.\n');
    close(fig);
end

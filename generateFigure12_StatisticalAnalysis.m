function generateFigure12_StatisticalAnalysis(results, baselineResults, outputDir)
% FIGURE 12: Statistical analysis
%   Subfigures: (a) Box plots, (b) Violin-like distribution, (c) Calibration

    fig = figure('Position', [100 100 1200 400], 'Color', 'w');

    % (a) Box plots of accuracy across folds
    subplot(1, 3, 1);
    methodNames = fieldnames(baselineResults);
    topMethods = {'GradientBoosting', 'RandomForest', 'SVM_RBF'};
    allAccData = [];
    groupLabels = {};
    for m = 1:length(topMethods)
        if isfield(baselineResults, topMethods{m})
            accData = baselineResults.(topMethods{m}).accuracy * 100;
            allAccData = [allAccData; accData];
            groupLabels = [groupLabels; repmat(topMethods(m), length(accData), 1)];
        end
    end
    allAccData = [allAccData; results.accuracy * 100];
    groupLabels = [groupLabels; repmat({'FQI-DNN-XAI'}, length(results.accuracy), 1)];

    boxplot(allAccData, groupLabels, 'Colors', 'brgk', 'Widths', 0.5);
    ylabel('Accuracy (%)', 'FontSize', 11);
    title('(a) Accuracy Distribution (10-Fold)', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 9);

    % (b) Score distribution
    subplot(1, 3, 2);
    posScores = results.allScores(results.allTrue == 1);
    negScores = results.allScores(results.allTrue == 0);
    histogram(posScores, 20, 'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.6, ...
              'DisplayName', 'IHD Positive'); hold on;
    histogram(negScores, 20, 'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.6, ...
              'DisplayName', 'IHD Negative');
    xlabel('Predicted Probability', 'FontSize', 11);
    ylabel('Count', 'FontSize', 11);
    title('(b) Prediction Score Distribution', 'FontSize', 11, 'FontWeight', 'bold');
    legend('FontSize', 9); set(gca, 'FontSize', 9);

    % (c) Calibration curve
    subplot(1, 3, 3);
    nBins = 10;
    binEdges = linspace(0, 1, nBins + 1);
    meanPred = zeros(1, nBins);
    fracPos = zeros(1, nBins);
    for b = 1:nBins
        inBin = results.allScores >= binEdges(b) & results.allScores < binEdges(b+1);
        if sum(inBin) > 0
            meanPred(b) = mean(results.allScores(inBin));
            fracPos(b) = mean(results.allTrue(inBin));
        end
    end
    plot([0 1], [0 1], 'k--', 'LineWidth', 1); hold on;
    plot(meanPred, fracPos, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, ...
         'MarkerFaceColor', 'r');
    xlabel('Mean Predicted Probability', 'FontSize', 11);
    ylabel('Fraction of Positives', 'FontSize', 11);
    title('(c) Calibration Curve', 'FontSize', 11, 'FontWeight', 'bold');
    legend({'Perfect', 'FQI-DNN-XAI'}, 'FontSize', 9, 'Location', 'southeast');
    grid on; set(gca, 'FontSize', 9);

    sgtitle('Figure 12. Statistical Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure12_StatisticalAnalysis.fig'));
    saveas(fig, fullfile(outputDir, 'Figure12_StatisticalAnalysis.png'));
    fprintf('  Figure 12: Statistical analysis saved.\n');
    close(fig);
end

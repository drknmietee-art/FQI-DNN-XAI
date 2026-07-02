function generateFigure5_SHAPAnalysis(shapValues, X_samples, featureNames, outputDir)
% FIGURE 5: SHAP explainability analysis
%   Subfigures: (a) Summary beeswarm, (b) Dependence plot, (c) Force plot

    % Map SHAP values back to original features (aggregate from quantum space)
    d_orig = length(featureNames);
    d_shap = size(shapValues, 2);

    if d_shap > d_orig
        % Aggregate: mean absolute SHAP per original feature
        shapOrig = zeros(size(shapValues, 1), d_orig);
        ratio = floor(d_shap / d_orig);
        for j = 1:d_orig
            cols = min((j-1)*ratio+1, d_shap):min(j*ratio, d_shap);
            if ~isempty(cols)
                shapOrig(:, j) = mean(shapValues(:, cols), 2);
            end
        end
    else
        shapOrig = shapValues(:, 1:min(d_shap, d_orig));
        if d_shap < d_orig
            shapOrig(:, d_shap+1:d_orig) = 0;
        end
    end

    fig = figure('Position', [100 100 1200 400], 'Color', 'w');

    % (a) Summary bar plot (mean |SHAP|)
    subplot(1, 3, 1);
    meanSHAP = mean(abs(shapOrig), 1);
    [sortedSHAP, sortIdx] = sort(meanSHAP, 'descend');
    barh(sortedSHAP(end:-1:1), 'FaceColor', [0.2 0.5 0.8]);
    yticks(1:d_orig);
    yticklabels(featureNames(sortIdx(end:-1:1)));
    xlabel('Mean |SHAP Value|', 'FontSize', 11);
    title('(a) SHAP Feature Importance', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9);

    % (b) SHAP dependence plot for top feature
    subplot(1, 3, 2);
    topFeat = sortIdx(1);
    if size(X_samples, 2) >= topFeat
        scatter(X_samples(:, topFeat), shapOrig(:, topFeat), 20, ...
                shapOrig(:, topFeat), 'filled', 'MarkerFaceAlpha', 0.6);
        colorbar; colormap(gca, cool);
        xlabel(featureNames{topFeat}, 'FontSize', 11);
        ylabel(sprintf('SHAP(%s)', featureNames{topFeat}), 'FontSize', 11);
        title(sprintf('(b) Dependence: %s', featureNames{topFeat}), ...
              'FontSize', 12, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 9);

    % (c) Force plot for one sample
    subplot(1, 3, 3);
    sampleIdx = 1;
    shapSample = shapOrig(sampleIdx, :);
    [sortedS, sIdx] = sort(shapSample);
    colors = zeros(d_orig, 3);
    for j = 1:d_orig
        if sortedS(j) >= 0
            colors(j, :) = [0.8 0.2 0.2]; % Positive = red
        else
            colors(j, :) = [0.2 0.4 0.8]; % Negative = blue
        end
    end
    barh(sortedS, 'FaceColor', 'flat', 'CData', colors);
    yticks(1:d_orig);
    yticklabels(featureNames(sIdx));
    xlabel('SHAP Value', 'FontSize', 11);
    title('(c) Force Plot (Sample 1)', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9);

    sgtitle('Figure 5. SHAP Explainability Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure5_SHAPAnalysis.fig'));
    saveas(fig, fullfile(outputDir, 'Figure5_SHAPAnalysis.png'));
    fprintf('  Figure 5: SHAP analysis saved.\n');
    close(fig);
end

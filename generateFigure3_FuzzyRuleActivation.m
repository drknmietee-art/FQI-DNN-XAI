function generateFigure3_FuzzyRuleActivation(firingStrengths, normalizedFS, y, outputDir)
% FIGURE 3: Fuzzy rule activation patterns
%   Subfigures: (a) Heatmap, (b) Distribution by class

    fig = figure('Position', [100 100 1000 450], 'Color', 'w');

    % (a) Heatmap of firing strengths (show first 50 samples, 20 rules)
    subplot(1, 2, 1);
    nShow = min(50, size(normalizedFS, 1));
    rShow = min(20, size(normalizedFS, 2));
    imagesc(normalizedFS(1:nShow, 1:rShow));
    colorbar; colormap(hot);
    xlabel('Rule Index', 'FontSize', 11);
    ylabel('Sample Index', 'FontSize', 11);
    title('(a) Rule Firing Strengths', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'FontSize', 10);

    % (b) Distribution of dominant rules
    subplot(1, 2, 2);
    [~, dominantRules] = max(normalizedFS, [], 2);
    idxPos = y == 1;
    idxNeg = y == 0;
    edges = 1:max(dominantRules)+1;
    histogram(dominantRules(idxPos), edges, 'FaceColor', [0.8 0.2 0.2], ...
              'FaceAlpha', 0.6, 'DisplayName', 'IHD Positive'); hold on;
    histogram(dominantRules(idxNeg), edges, 'FaceColor', [0.2 0.4 0.8], ...
              'FaceAlpha', 0.6, 'DisplayName', 'IHD Negative');
    xlabel('Dominant Rule', 'FontSize', 11);
    ylabel('Count', 'FontSize', 11);
    title('(b) Dominant Rules by Class', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 9);
    set(gca, 'FontSize', 10);

    sgtitle('Figure 3. Fuzzy Rule Activation Patterns', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure3_FuzzyRuleActivation.fig'));
    saveas(fig, fullfile(outputDir, 'Figure3_FuzzyRuleActivation.png'));
    fprintf('  Figure 3: Fuzzy rule activation saved.\n');
    close(fig);
end

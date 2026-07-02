function generateFigure2_MembershipFunctions(X, centers, spreads, featureNames, outputDir)
% FIGURE 2: Gaussian membership functions for selected clinical features
%   Subfigures: (a) Age, (b) Resting BP, (c) Cholesterol, (d) Max HR

    selectedFeatures = [1, 4, 5, 8]; % age, trestbps, chol, thalach
    subLabels = {'(a) Age', '(b) Resting Blood Pressure', ...
                 '(c) Serum Cholesterol', '(d) Maximum Heart Rate'};
    linguisticTerms = {'Low', 'Medium', 'High'};
    colors = [0.2 0.4 0.8; 0.2 0.7 0.3; 0.8 0.2 0.2];

    fig = figure('Position', [100 100 900 700], 'Color', 'w');
    for idx = 1:4
        j = selectedFeatures(idx);
        subplot(2, 2, idx);

        x_range = linspace(0, 1, 200);
        for l = 1:3
            mu = exp(-((x_range - centers(j, l)).^2) ./ (2 * spreads(j, l)^2));
            plot(x_range, mu, 'Color', colors(l, :), 'LineWidth', 2); hold on;
        end

        xlabel(featureNames{j}, 'FontSize', 11);
        ylabel('Membership Degree \mu', 'FontSize', 11);
        title(subLabels{idx}, 'FontSize', 12, 'FontWeight', 'bold');
        legend(linguisticTerms, 'Location', 'best', 'FontSize', 9);
        grid on; ylim([0 1.05]);
        set(gca, 'FontSize', 10);
    end
    sgtitle('Figure 2. Gaussian Membership Functions', 'FontSize', 14, 'FontWeight', 'bold');

    saveas(fig, fullfile(outputDir, 'Figure2_MembershipFunctions.fig'));
    saveas(fig, fullfile(outputDir, 'Figure2_MembershipFunctions.png'));
    fprintf('  Figure 2: Membership functions saved.\n');
    close(fig);
end
